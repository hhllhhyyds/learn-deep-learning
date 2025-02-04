#[cfg(test)]
use candle_core::{Device, Tensor};

#[test]
fn show_array() {
    let arr_i64 = Tensor::arange::<i64>(0, 12, &Device::Cpu).unwrap();
    println!("array i64 element count = {}", arr_i64.elem_count());

    let arr_f64 = Tensor::arange::<f64>(0., 12., &Device::new_cuda(0).unwrap()).unwrap();
    println!("array f64 element count = {}", arr_f64.elem_count());

    println!("array f64 shape = {:#?}", arr_f64.shape());

    let arr_f64_reshaped = arr_f64.reshape((4, 3)).unwrap();
    println!("array f64 reshaped shape = {:#?}", arr_f64_reshaped.shape());
}

#[test]
fn all_zero_one_array() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let zero_arr = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &gpu_device).unwrap();
    println!("zero arr = \n{}", zero_arr);

    let one_arr = Tensor::ones((2, 3, 4), candle_core::DType::F32, &gpu_device).unwrap();
    println!("one arr = \n{}", one_arr);
}

#[test]
fn randn_array() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let randn_arr = Tensor::randn(0., 1., (3, 4), &gpu_device).unwrap();
    println!("randn arr = \n{}", randn_arr);
}

#[test]
fn exact_array() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let exact_arr =
        Tensor::new(&[[2i64, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]], &gpu_device).unwrap();
    println!("exact arr = \n{}", exact_arr);
}

#[test]
fn tensor_index_slice() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let tensor = Tensor::arange::<f64>(0., 12., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    println!(
        "X[-1] = {}",
        tensor.get(tensor.dim(0).unwrap() - 1).unwrap()
    );

    println!(
        "X[1:3] = \n{}",
        tensor
            .index_select(&Tensor::arange(1u32, 3, &gpu_device).unwrap(), 0)
            .unwrap()
    );
}

#[test]
fn unary_ops() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let tensor = Tensor::arange::<f64>(0., 12., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    println!("X.exp() = {}", tensor.exp().unwrap());
}

#[test]
fn binary_ops() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let x = Tensor::arange::<f64>(0., 12., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    let y = Tensor::arange::<f64>(-12., 0., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    println!("x + y = \n{}", (&x + &y).unwrap());
    println!("x - y = \n{}", (&x - &y).unwrap());
    println!("x * y = \n{}", (&x * &y).unwrap());
    println!("x / y = \n{}", (&x / &y).unwrap());

    println!("x > y = \n{}", (x.gt(&y)).unwrap());
}

#[test]
fn tensor_concat() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let x = Tensor::arange::<f64>(0., 12., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    let y = Tensor::arange::<f64>(-12., 0., &gpu_device)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    println!(
        "Tensor::cat(&[&x, &y], 0) = \n{}",
        Tensor::cat(&[&x, &y], 0).unwrap()
    );

    println!(
        "Tensor::cat(&[&x, &y], 1) = \n{}",
        Tensor::cat(&[&x, &y], 1).unwrap()
    );

    println!("x.sum() = {}", x.sum_all().unwrap());
}

#[test]
fn broadcast_ops() {
    let gpu_device = Device::new_cuda(0).unwrap();

    let x = Tensor::arange::<f64>(1., 4., &gpu_device)
        .unwrap()
        .reshape((3, 1))
        .unwrap();

    let y = Tensor::arange::<f64>(1., 3., &gpu_device)
        .unwrap()
        .reshape((1, 2))
        .unwrap();

    println!("x = \n{}", x);
    println!("y = \n{}", y);

    let x_bc_add_y = x.broadcast_add(&y).unwrap();
    println!("x broadcast add y = \n{}", x_bc_add_y);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
