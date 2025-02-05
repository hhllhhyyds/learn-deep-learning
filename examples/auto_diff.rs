use candle_core::{Device, Tensor, Var};

#[test]
fn randn_grad() {
    let device = &Device::new_cuda(0).unwrap();

    let x = Var::randn(-2.5f64, 1., 4, &device).unwrap();
    println!("x =\n{x}");

    fn f(t: &Tensor) -> Tensor {
        let mut b = (t * 2.0).unwrap();
        while b
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
            < 1000.
        {
            b = (b * 2.0).unwrap();
        }
        let c = if b.sum_all().unwrap().to_scalar::<f64>().unwrap() > 0. {
            b
        } else {
            (b * 100.).unwrap()
        };

        c
    }

    let y = f(&x);
    let grad = y.backward().unwrap();

    println!("grad = \n{}", grad.get(&x).unwrap());
    println!("y / x = \n{}", (&y / x.as_tensor()).unwrap());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = &Device::new_cuda(0)?;

    let x = Var::from_tensor(&Tensor::arange(0.0_f32, 4.0, device)?.reshape(4)?)?;
    println!("x =\n{x}");

    let y = ((x.as_tensor() * x.as_tensor())?.sum_all()? * 2.0)?;
    println!("y =\n{y}");

    let y_grads = y.backward()?;
    println!("y grads =\n{}", y_grads.get(&x).unwrap());

    let y = x.sum_all()?;
    println!("y =\n{y}");

    let y_grads = y.backward()?;
    println!("y grads =\n{}", y_grads.get(&x).unwrap());

    let y = (x.as_tensor() * x.as_tensor())?;
    println!("y =\n{y}");

    let y_grads = y.backward()?;
    println!("y grads =\n{}", y_grads.get(&x).unwrap());

    let u = y.clone();
    println!("u =\n{u}");

    let z = (u * x.as_tensor())?;
    let z_sum_grads = z.sum_all()?.backward()?;
    println!("z sum grads =\n{}", z_sum_grads.get(&x).unwrap());

    let u = y.detach();
    println!("u =\n{u}");

    let z = (u * x.as_tensor())?;
    let z_sum_grads = z.sum_all()?.backward()?;
    println!("z sum grads =\n{}", z_sum_grads.get(&x).unwrap());

    Ok(())
}
