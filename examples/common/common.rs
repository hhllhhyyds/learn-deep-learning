use plotpy::{Curve, Plot};

pub fn plot_progress(train_progress: &[(usize, f32)], val_progress: &[(usize, f32)]) {
    fn batch_to_epoch(batch_progress: &[(usize, f32)]) -> Vec<(f32, f32)> {
        let max_batch = batch_progress.iter().map(|p| p.0).max().unwrap() + 1;

        let mut epoch_progress = Vec::new();
        let mut epoch = -1;

        for p in batch_progress {
            if p.0 == 0 {
                epoch += 1;
            }
            epoch_progress.push((epoch as f32 + p.0 as f32 / max_batch as f32, p.1));
        }

        epoch_progress
    }

    fn add_points_to_curve(curve: &mut Curve, points: &[(f32, f32)]) {
        curve.points_begin();
        points.iter().for_each(|p| {
            curve.points_add(p.0, p.1);
        });
        curve.points_end();
    }

    let train_progress = batch_to_epoch(train_progress);
    let val_progress = batch_to_epoch(val_progress);

    let mut train_curve = Curve::new();
    train_curve.set_line_color("red");
    train_curve.set_line_width(0.5);
    train_curve.set_label("training loss");
    add_points_to_curve(&mut train_curve, &train_progress);

    let mut val_curve = Curve::new();
    val_curve.set_line_color("blue");
    val_curve.set_line_width(0.5);
    val_curve.set_label("validation loss");
    add_points_to_curve(&mut val_curve, &val_progress);

    let mut plot = Plot::new();
    plot.add(&train_curve).set_labels("epoch", "loss");
    plot.add(&val_curve);
    plot.legend();

    plot.show("./target/train_progress.svg").unwrap();
}
