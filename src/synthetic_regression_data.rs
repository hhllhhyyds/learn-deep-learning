use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use rand::{seq::SliceRandom, thread_rng};

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
        let shape = weights.shape();
        assert!(shape.dims().to_vec() == vec![weights.elem_count(), 1]);

        Self {
            weights,
            bias: f32::default(),
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
        let d = self.weights.shape().dim(0)?;
        let n = self.sample_count();

        let samples = Tensor::randn(0f32, 1f32, (n, d), &self.device)?;
        let noise = (Tensor::randn(0f32, 1f32, (n, 1), &self.device)? * self.noise as f64)?;
        let targets = (self.bias as f64 + (samples.matmul(&self.weights)? + noise)?)?;

        Ok(SyntheticRegressionData {
            samples,
            targets,
            paras: self.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticRegressionData {
    samples: Tensor,
    targets: Tensor,
    paras: SyntheticRegressionDataBuilder,
}

pub struct DatasetRandomIter {
    indices: Vec<usize>,
    current_index: usize,
    data: SyntheticRegressionData,
}

impl Iterator for DatasetRandomIter {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.indices.len() {
            None
        } else {
            let t = (
                self.data.samples.get(self.indices[self.current_index]).ok(),
                self.data.targets.get(self.indices[self.current_index]).ok(),
            );
            self.current_index += 1;
            t.0.zip(t.1)
        }
    }
}

impl SyntheticRegressionData {
    pub fn iter(&self, is_train: bool) -> DatasetRandomIter {
        let mut indices = if is_train {
            0..self.paras.num_train
        } else {
            self.paras.num_train..self.paras.sample_count()
        }
        .collect::<Vec<_>>();
        indices.shuffle(&mut thread_rng());

        DatasetRandomIter {
            indices,
            current_index: 0,
            data: self.clone(),
        }
    }

    pub fn weights(&self) -> Tensor {
        self.paras.weights.clone()
    }

    pub fn bias(&self) -> f32 {
        self.paras.bias
    }
}
