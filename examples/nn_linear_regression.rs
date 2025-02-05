use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_datasets::Batcher;

use candle_nn::{Module, Optimizer, VarBuilder, VarMap, SGD};
use learn_deep_learning::synthetic_regression_data::SyntheticRegressionDataBuilder;

#[path = "common/common.rs"]
pub mod common;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let max_epoch = args.get(1).unwrap_or(&"3".to_string()).parse::<usize>()?;

    let device = Arc::new(Device::new_cuda(0)?);

    let data =
        SyntheticRegressionDataBuilder::new(Tensor::from_slice(&[2f32, -3.4], (2, 1), &device)?)
            .device(device.clone())
            .bias(4.2)
            .build()?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = candle_nn::linear(2, 1, vb.pp("linear"))?;
    let mut opt = SGD::new(varmap.all_vars(), 0.03)?;

    let mut train_progress = Vec::new();
    let mut val_progress = Vec::new();
    for _ in 0..max_epoch {
        let training_batcher = Batcher::new2(data.iter(true))
            .batch_size(32)
            .return_last_incomplete_batch(false);
        let validate_batcher = Batcher::new2(data.iter(false))
            .batch_size(32)
            .return_last_incomplete_batch(false);

        for (i, batch) in training_batcher.enumerate() {
            let (features, target) = batch?;
            let loss = &model.forward(&features)?.sub(&target)?.sqr()?.mean_all()?;
            opt.backward_step(loss)?;

            train_progress.push((i, loss.to_scalar::<f32>()?));
        }

        for (i, batch) in validate_batcher.enumerate() {
            let (features, target) = batch?;
            let loss = &model.forward(&features)?.sub(&target)?.sqr()?.mean_all()?;

            val_progress.push((i, loss.to_scalar::<f32>()?));
        }

        println!(
            "error in estimating weights:\n{}",
            (data.weights().clone().reshape((1, 2)) - model.weight())?
        );
        println!(
            "error in estimating bias:\n{}",
            (data.bias() as f64 - model.bias().unwrap())?
        );
    }

    common::plot_progress(&train_progress, &val_progress);

    Ok(())
}
