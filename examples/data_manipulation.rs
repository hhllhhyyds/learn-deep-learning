use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpu = Device::Cpu;

    let g = Tensor::arange::<f32>(0., 12., &cpu)?;
    println!("cpu g = {g}");

    let g = g.reshape((3, 4))?;

    let gpu = Device::new_cuda(0)?;
    let x = Tensor::arange::<f32>(0., 12., &gpu)?;
    println!("metal x = {x}");

    println!("x element count = {}", x.elem_count());

    println!("x shape = {:?}", x.shape());

    let x = x.reshape((3, 4))?;

    println!("x after reshape is\n{}, shape is {:?}", x, x.shape());

    let zeros_tensor = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &cpu)?;
    println!("tensor zeros:\n{}", zeros_tensor);

    println!(
        "tensor ones:\n{}",
        Tensor::ones((2, 3, 4), candle_core::DType::F32, &cpu)?
    );

    println!("tensor random:\n{}", Tensor::randn(0.0, 1.0, (3, 4), &cpu)?);

    println!(
        "tensor specified:\n{}",
        Tensor::new(&[[2_i64, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]], &cpu)?
    );

    println!("x[-1] = {:?}", x.get(2)?.to_vec1::<f32>()?);
    println!(
        "x[1:3] = {:?}",
        x.index_select(&Tensor::new(&[1_i64, 2], &gpu)?, 0)?
            .to_vec2::<f32>()?
    );

    x.get(1)?.slice_set(&Tensor::new(&[17_f32], &gpu)?, 0, 2)?;
    println!("x = \n{}", x);

    let y = Tensor::from_slice(&[12_f32; 8], (2, 4), &gpu)?;
    let x = x.slice_assign(&[0..2, 0..4], &y)?;
    println!("x = \n{}", x);

    let z = x.to_device(&cpu)?;
    println!("x exp = \n{}", x.exp()?);
    println!("z exp = \n{}", z.exp()?);

    let p = Tensor::from_slice(&[1_f32, 2., 4., 8.], (1, 4), &gpu)?;
    let q = Tensor::from_slice(&[2_f32; 4], (1, 4), &gpu)?;

    println!("p = {p},\nq = {q}");

    println!("p + q = {}", (p.clone() + q.clone())?);
    println!("p - q = {}", (p.clone() - q.clone())?);
    println!("p * q = {}", (p.clone() * q.clone())?);
    println!("p / q = {}", (p.clone() / q.clone())?);
    println!("p ** q = {}", (p.clone().pow(&q))?);

    let gz0 = Tensor::cat(&[g.clone(), z.clone()], 0)?;
    let gz1 = Tensor::cat(&[g.clone(), z.clone()], 1)?;

    println!("gz0 = \n{gz0}");
    println!("gz1 = \n{gz1}");

    println!("z == g:\n{}", z.eq(&g)?);

    println!("z < g:\n{}", z.lt(&g)?);

    println!("g sum = {}", g.sum_all()?);

    let a = Tensor::arange(0_i64, 3, &gpu)?.reshape((3, 1))?;
    println!("a = \n{a}");

    let b = Tensor::arange(0_i64, 2, &gpu)?.reshape((1, 2))?;
    println!("b = \n{b}");

    println!("a + b = \n{}", a.broadcast_add(&b)?);

    Ok(())
}
