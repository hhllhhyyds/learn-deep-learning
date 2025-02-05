#[cfg(test)]
use candle_core::{Device, Tensor};

#[test]
fn tensor_sum_mean() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let x = Tensor::arange::<f64>(0., 12., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    let x_axis_sum = x.sum(1).unwrap();

    println!("x axis sum = \n{}", x_axis_sum);

    let y_axis_sum = x.sum(0).unwrap();

    println!("y axis sum = \n{}", y_axis_sum);

    let two_axis_sum = x.sum((0, 1)).unwrap();

    println!("two axis sum = \n{}", two_axis_sum);

    let y_axis_mean = x.mean(0).unwrap();

    println!("y axis mean = \n{}", y_axis_mean);

    let x_axis_sum_keep_dim = x.sum_keepdim(1).unwrap();

    println!("x axis sum keep dim = \n{}", x_axis_sum_keep_dim);
}

#[test]
fn tensor_mat_mul() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let x = Tensor::arange::<f64>(0., 12., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    let y = Tensor::ones((4, 5), candle_core::DType::F64, &gpu_device).unwrap();

    let z = x.matmul(&y).unwrap();

    println!("x mat mul y = \n{}", z);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
