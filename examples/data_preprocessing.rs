use std::io::Write;

use polars::prelude::*;

use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::path::Path::new("..").join("data");
    if !std::fs::exists(data_dir.clone())? {
        std::fs::create_dir(data_dir.clone())?;
    }

    let data_file = data_dir.join("house_tiny.csv");
    let mut buffer = std::fs::File::create(data_file.clone())?;
    write!(
        buffer,
        r#"NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000"#
    )?;
    buffer.sync_all()?;

    let df1 = polars::prelude::CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(data_file.into())?
        .finish()?;

    println!("{}", df1);

    let df2 = df1
        .clone()
        .lazy()
        .select([
            when(col("NumRooms").eq(lit("NA")))
                .then(lit(NULL))
                .otherwise(col("NumRooms"))
                .cast(DataType::Float32)
                .alias("NumRooms"),
            when(col("RoofType").eq(lit("NA")))
                .then(lit(NULL))
                .otherwise(col("RoofType"))
                .alias("RoofType"),
            col("Price").cast(DataType::Float32),
        ])
        .collect()?;
    println!("{}", df2);

    let inputs = df2
        .clone()
        .lazy()
        .select([
            col("NumRooms")
                .fill_null(col("NumRooms").mean())
                .alias("NumRooms"),
            when(col("RoofType").is_null())
                .then(0)
                .otherwise(1)
                .cast(DataType::Float32)
                .alias("RoofType_Slate"),
            when(col("RoofType").is_null())
                .then(1)
                .otherwise(0)
                .cast(DataType::Float32)
                .alias("RoofType_Nan"),
        ])
        .collect()?;
    println!("inputs:\n{}", inputs);

    let target = df2.clone().lazy().select([col("Price")]).collect()?;
    println!("target:\n{}", target);

    let cpu = Device::Cpu;
    let input_buffer = inputs.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let x_tensor = Tensor::from_slice(input_buffer.as_slice().unwrap(), inputs.shape(), &cpu)?;
    let y_tensor = Tensor::from_slice(
        target
            .to_ndarray::<Float32Type>(IndexOrder::C)?
            .as_slice()
            .unwrap(),
        target.shape(),
        &cpu,
    )?
    .transpose(0, 1)?;

    println!("x tensor:\n{x_tensor}");
    println!("y tensor:\n{y_tensor}");

    Ok(())
}
