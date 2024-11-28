use candle_core::{Device, Tensor, Var};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let device = &Device::Cpu;
    let device = &Device::new_metal(0)?;

    let x = Var::from_tensor(&Tensor::arange(0.0_f32, 4.0, device)?.reshape(4)?)?;

    let y = (x.as_tensor() * x.as_tensor())?;
    println!("y =\n{y}");

    let y_grads = y.backward()?;
    println!("y grads =\n{:?}", y_grads);

    let y_grad_x = y_grads.get(x.as_tensor()).unwrap();
    println!("y grad x =\n{}", y_grad_x);

    let z = (x.as_tensor() * x.as_tensor())?.sum_all()?;
    println!("z =\n{z}");

    let z_grads = z.backward()?;
    println!("z grads =\n{:?}", z_grads);

    let z_grad_x = z_grads.get(x.as_tensor()).unwrap();
    println!("z grad x =\n{}", z_grad_x);

    let u = y.clone();
    let z = (x.as_tensor() * u.clone())?.sum_all()?;
    let grads = z.backward()?;
    let grad = grads.get(x.as_tensor()).unwrap();
    println!("grad =\n{}", grad);

    let u = y.detach();
    let z = (x.as_tensor() * u.clone())?.sum_all()?;
    let grads = z.backward()?;
    let grad = grads.get(x.as_tensor()).unwrap();
    println!("{}", grad.eq(&u)?);

    Ok(())
}
