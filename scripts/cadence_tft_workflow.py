"""
Temporal Fusion Transformer (TFT) Workflow for Bakery Sales Forecasting
Uses Cadence for workflow orchestration
"""

import asyncio
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch


# Activity Definitions
@activity.defn
async def load_and_prepare_data() -> dict:
    """Load and prepare bakery sales data for TFT"""
    activity.logger.info("Loading training and test data...")

    # Load data
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

    # Combine for preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Convert date to datetime
    df['Datum'] = pd.to_datetime(df['Datum'])
    df = df.sort_values(['Warengruppe', 'Datum']).reset_index(drop=True)

    # Create time index
    df['time_idx'] = (df['Datum'] - df['Datum'].min()).dt.days

    # Add temporal features
    df['day_of_week'] = df['Datum'].dt.dayofweek
    df['day_of_month'] = df['Datum'].dt.day
    df['week_of_year'] = df['Datum'].dt.isocalendar().week
    df['month'] = df['Datum'].dt.month
    df['year'] = df['Datum'].dt.year

    # Create group ID
    df['group_id'] = df['Warengruppe'].astype(str)

    activity.logger.info(f"Data prepared: {len(df)} rows, time range: {df['time_idx'].min()} to {df['time_idx'].max()}")

    return {
        'data': df.to_dict('records'),
        'max_time_idx': int(df['time_idx'].max()),
        'train_cutoff': int(df[df['year'] < 2018]['time_idx'].max())
    }


@activity.defn
async def train_tft_model(data_dict: dict) -> dict:
    """Train Temporal Fusion Transformer model"""
    activity.logger.info("Starting TFT model training...")

    # Reconstruct dataframe
    df = pd.DataFrame(data_dict['data'])
    train_cutoff = data_dict['train_cutoff']

    # Define maximum prediction length
    max_prediction_length = 30  # Forecast 30 days ahead
    max_encoder_length = 90     # Use 90 days of history

    # Create TimeSeriesDataSet for training
    training = TimeSeriesDataSet(
        df[df['time_idx'] <= train_cutoff],
        time_idx='time_idx',
        target='Umsatz',
        group_ids=['group_id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['group_id'],
        time_varying_known_categoricals=['day_of_week', 'month'],
        time_varying_known_reals=['time_idx', 'day_of_month', 'week_of_year'],
        time_varying_unknown_reals=['Umsatz'],
        target_normalizer=GroupNormalizer(
            groups=['group_id'], transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    # Create dataloaders
    batch_size = 64
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    activity.logger.info("Creating TFT model...")

    # Configure TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )

    activity.logger.info(f"Model parameters: {tft.size()/1e3:.1f}k")

    # Configure trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        limit_train_batches=50,
        enable_checkpointing=True,
    )

    activity.logger.info("Training model...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Save model
    best_model_path = "tft_bakery_model.ckpt"
    trainer.save_checkpoint(best_model_path)

    activity.logger.info(f"Model training completed. Saved to {best_model_path}")

    return {
        'model_path': best_model_path,
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', 0.0))
    }


@activity.defn
async def make_predictions(data_dict: dict, model_info: dict) -> dict:
    """Generate predictions using trained TFT model"""
    activity.logger.info("Generating predictions...")

    # Reconstruct dataframe
    df = pd.DataFrame(data_dict['data'])
    train_cutoff = data_dict['train_cutoff']

    max_prediction_length = 30
    max_encoder_length = 90

    # Recreate training dataset structure
    training = TimeSeriesDataSet(
        df[df['time_idx'] <= train_cutoff],
        time_idx='time_idx',
        target='Umsatz',
        group_ids=['group_id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['group_id'],
        time_varying_known_categoricals=['day_of_week', 'month'],
        time_varying_known_reals=['time_idx', 'day_of_month', 'week_of_year'],
        time_varying_unknown_reals=['Umsatz'],
        target_normalizer=GroupNormalizer(
            groups=['group_id'], transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Load best model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_info['model_path'])

    # Generate predictions on test set
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )

    # Make predictions
    predictions = best_tft.predict(test_dataloader, return_x=True)

    activity.logger.info(f"Predictions generated: shape {predictions.output.shape}")

    # Calculate metrics on test set (2018 data)
    test_df = df[df['time_idx'] > train_cutoff].copy()

    # Get actual values and predictions for comparison
    actual_values = test_df['Umsatz'].values[:len(predictions.output)]
    predicted_values = predictions.output.numpy().flatten()[:len(actual_values)]

    mae = np.mean(np.abs(actual_values - predicted_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))

    activity.logger.info(f"Test Metrics - MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")

    return {
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'num_predictions': len(predicted_values)
    }


# Workflow Definition
@workflow.defn
class BakeryTFTWorkflow:
    """Main workflow for bakery sales forecasting with TFT"""

    @workflow.run
    async def run(self) -> dict:
        workflow.logger.info("Starting Bakery TFT Workflow...")

        # Step 1: Load and prepare data
        data_dict = await workflow.execute_activity(
            load_and_prepare_data,
            start_to_close_timeout=timedelta(minutes=5),
        )
        workflow.logger.info(f"Data loaded: {data_dict['max_time_idx']} time steps")

        # Step 2: Train TFT model
        model_info = await workflow.execute_activity(
            train_tft_model,
            data_dict,
            start_to_close_timeout=timedelta(minutes=30),
        )
        workflow.logger.info(f"Model trained with val_loss: {model_info['best_val_loss']:.4f}")

        # Step 3: Generate predictions
        predictions = await workflow.execute_activity(
            make_predictions,
            args=[data_dict, model_info],
            start_to_close_timeout=timedelta(minutes=10),
        )
        workflow.logger.info(f"Predictions complete - MAE: {predictions['mae']:.2f}")

        return {
            'status': 'completed',
            'model_path': model_info['model_path'],
            'validation_loss': model_info['best_val_loss'],
            'test_metrics': predictions
        }


async def main():
    """Main entry point to run the workflow"""
    # Connect to Cadence server
    client = await Client.connect("localhost:7233")

    # Run worker with our workflow and activities
    async with Worker(
        client,
        task_queue="bakery-tft-task-queue",
        workflows=[BakeryTFTWorkflow],
        activities=[load_and_prepare_data, train_tft_model, make_predictions],
    ):
        # Execute workflow
        result = await client.execute_workflow(
            BakeryTFTWorkflow.run,
            id="bakery-tft-workflow-001",
            task_queue="bakery-tft-task-queue",
        )

        print("\n" + "="*60)
        print("BAKERY TFT WORKFLOW COMPLETED")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Model Path: {result['model_path']}")
        print(f"Validation Loss: {result['validation_loss']:.4f}")
        print(f"\nTest Set Metrics (2018):")
        print(f"  MAE:  {result['test_metrics']['mae']:.2f}")
        print(f"  MAPE: {result['test_metrics']['mape']:.2f}%")
        print(f"  RMSE: {result['test_metrics']['rmse']:.2f}")
        print(f"  Predictions: {result['test_metrics']['num_predictions']}")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
