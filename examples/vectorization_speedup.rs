use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);

    let cpu = &Device::Cpu;
    let gpu = &Device::new_metal(0)?;

    let n = args[1].parse::<usize>()?;
    let a_arr = (n..(2 * n)).map(|i| i as f32).collect::<Vec<f32>>();
    let b_arr = ((n * 2)..(n * 3)).map(|i| i as f32).collect::<Vec<f32>>();
    let mut c_arr = vec![0f32; n];

    let a_cpu = Tensor::arange(0f32, n as f32, cpu)?;
    let b_cpu = Tensor::arange(1f32, (n + 1) as f32, cpu)?;

    let a_gpu = Tensor::arange(0f32, n as f32, gpu)?;
    let b_gpu = Tensor::arange(1f32, (n + 1) as f32, gpu)?;

    let mut start = std::time::Instant::now();
    for i in 0..n {
        c_arr[i] = a_arr[i] + b_arr[i];
    }
    println!(
        "loop time = {}",
        (std::time::Instant::now() - start).as_secs_f32()
    );

    start = std::time::Instant::now();
    let c_cpu = (a_cpu + b_cpu)?;
    println!(
        "cpu tensor time = {}",
        (std::time::Instant::now() - start).as_secs_f32()
    );
    println!("sum = {}", c_cpu.sum_all()?);

    start = std::time::Instant::now();
    let c_gpu = (a_gpu + b_gpu)?;
    println!(
        "gpu tensor time = {}",
        (std::time::Instant::now() - start).as_secs_f32()
    );
    println!("sum = {}", c_gpu.sum_all()?);

    Ok(())
}
