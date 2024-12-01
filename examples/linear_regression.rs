use std::sync::Arc;

use candle_core::{Device, Error, Result, Tensor, Var};
use candle_datasets::Batcher;

use rand::{seq::SliceRandom, thread_rng};

pub struct RegressionDatasetRandomIter {
    indices: Vec<usize>,
    current_index: usize,
    features: Tensor,
    targets: Tensor,
}

impl RegressionDatasetRandomIter {
    pub fn randomize_indices(&mut self) {
        self.indices.shuffle(&mut thread_rng());
    }

    pub fn next_tensor(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_index == self.indices.len() {
            None
        } else {
            let t = (
                self.features.get(self.indices[self.current_index]).ok(),
                self.targets.get(self.indices[self.current_index]).ok(),
            );
            self.current_index += 1;
            t.0.zip(t.1)
        }
    }
}

impl Iterator for RegressionDatasetRandomIter {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == 0 {
            self.randomize_indices();
        }
        self.next_tensor()
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticRegressionData {
    pub features: Tensor,
    pub targets: Tensor,
    pub paras: SyntheticRegressionDataBuilder,
}

impl SyntheticRegressionData {
    pub fn iter(&self, is_train: bool) -> RegressionDatasetRandomIter {
        let indices = if is_train {
            0..self.paras.num_train
        } else {
            self.paras.num_train..self.paras.sample_count()
        }
        .collect::<Vec<_>>();

        RegressionDatasetRandomIter {
            indices,
            current_index: 0,
            features: self.features.clone(),
            targets: self.targets.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticRegressionDataBuilder {
    pub weights: Tensor,
    pub bias: f32,
    pub noise: f32,
    pub num_train: usize,
    pub num_validate: usize,
    pub device: Arc<Device>,
}

impl SyntheticRegressionDataBuilder {
    pub fn new(weights: Tensor) -> Self {
        Self {
            weights,
            bias: Default::default(),
            noise: 0.01,
            num_train: 1000,
            num_validate: 1000,
            device: Arc::new(Device::Cpu),
        }
    }

    pub fn device(mut self, device: Arc<Device>) -> Self {
        self.device = device;
        self
    }

    pub fn bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    pub fn noise(mut self, noise: f32) -> Self {
        self.noise = noise;
        self
    }

    pub fn num_train(mut self, num_train: usize) -> Self {
        self.num_train = num_train;
        self
    }

    pub fn num_validate(mut self, num_validate: usize) -> Self {
        self.num_validate = num_validate;
        self
    }

    pub fn sample_count(&self) -> usize {
        self.num_train + self.num_validate
    }

    pub fn build(&self) -> Result<SyntheticRegressionData> {
        let shape = self.weights.shape();
        assert!(shape.dims().to_vec() == vec![self.weights.elem_count(), 1]);

        let d = shape.dim(0)?;
        let n = self.sample_count();

        let features = Tensor::randn(0f32, 1f32, (n, d), &self.device)?;
        let noise = Tensor::randn(0f32, 1f32, (n, 1), &self.device)?;
        let targets = (self.bias as f64 + (features.matmul(&self.weights)? + noise)?)?;

        Ok(SyntheticRegressionData {
            features,
            targets,
            paras: self.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub weights: Var,
    pub bias: Var,
}

impl Model {
    pub fn new(features_count: usize, device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            weights: Var::randn(0_f32, 0.01, (features_count, 1), &device)?,
            bias: Var::zeros(1, candle_core::DType::F32, &device)?,
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        features.matmul(&self.weights)?.broadcast_add(&self.bias)
    }

    pub fn loss(prediction: &Tensor, targets: &Tensor) -> Result<Tensor> {
        ((prediction - targets)?.powf(2.0)? / 2_f64)?.mean_all()
    }

    pub fn step(&mut self, loss: &Tensor, learning_rate: f64) -> Result<()> {
        let grads = loss.backward()?;
        let weights_grads = grads
            .get(&self.weights)
            .ok_or(Error::debug("failed to get weights grad"))?;
        let bias_grads = grads
            .get(&self.bias)
            .ok_or(Error::debug("failed to get bias grad"))?;
        self.weights
            .set(&(self.weights.clone().as_tensor() - (learning_rate * weights_grads)?)?)?;
        self.bias
            .set(&(self.bias.as_tensor() - (learning_rate * bias_grads)?)?)?;

        Ok(())
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);

    let step_count = args[1].parse::<usize>()?;

    let device = Arc::new(Device::new_metal(0)?);

    let data =
        SyntheticRegressionDataBuilder::new(Tensor::from_slice(&[2f32, -3.4], (2, 1), &device)?)
            .device(device.clone())
            .bias(4.2)
            .build()?;

    println!("features shape = {:?}", data.features.shape());
    println!("targets shape = {:?}", data.targets.shape());

    let mut model = Model::new(2, device)?;

    let mut k = 0;
    for _ in 0..step_count {
        let training_batcher = Batcher::new2(data.iter(true))
            .batch_size(32)
            .return_last_incomplete_batch(false);
        let validate_batcher = Batcher::new2(data.iter(false))
            .batch_size(32)
            .return_last_incomplete_batch(false);

        for batch in training_batcher {
            let (features, target) = batch?;
            let loss = Model::loss(&model.forward(&features)?, &target)?;
            if k % 8 == 0 {
                println!(">>> training loss = {}", loss.to_scalar::<f32>()?);
            }
            k += 1;
            model.step(&loss, 0.03)?;
        }

        for batch in validate_batcher {
            let (features, target) = batch?;
            let loss = Model::loss(&model.forward(&features)?, &target)?;
            if k % 8 == 0 {
                println!("=== validate loss = {}", loss.to_scalar::<f32>()?);
            }
            k += 1;
        }

        println!(
            "error in estimating weights:\n{}",
            (data.paras.weights.clone() - model.weights.as_tensor())?
        );
        println!(
            "error in estimating bias:\n{}",
            (data.paras.bias as f64 - model.bias.as_tensor())?
        );
    }

    Ok(())
}
