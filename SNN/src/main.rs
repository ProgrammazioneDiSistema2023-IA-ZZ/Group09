use std::cmp::Ordering;

mod renderer;
mod network;
use network::{Snn};
use crate::network::{ExpectedOutput, Fault, Neuron, SignalInput, Unit};


#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {

    println!("Loading network...");

    let snn = Snn::from_numpy(vec!["./src/network_params/weights1.npy","./src/network_params/weights2.npy"], 0.9375, 1.0, 0.0, 0.0);

    println!("Loading inputs...");

    let inputs = SignalInput::from_numpy("./src/network_params/inputs.npy", Some(10));
    let expected_outputs = ExpectedOutput::from_numpy("./src/network_params/outputs.npy");

    println!("Starting...");

    let f = |neuron: &Neuron, input_signal: f32, current_step: i32, delta: f32, testing_add: Box<dyn Fn(f32, f32) -> f32>, testing_mul: Box<dyn Fn(f32, f32) -> f32>, testing_cmp: Box<dyn Fn(f32, f32) -> Ordering>| {

        let a = testing_add(neuron.potential, -neuron.rest_potential);
        let b = testing_mul(testing_add(current_step as f32, -neuron.last_activity as f32), delta);
        let c = (-b / neuron.time_constant).exp();
        let mut new_potential = testing_add(testing_add(neuron.rest_potential, testing_mul(a, c)), input_signal);
        let triggered = testing_cmp(new_potential, neuron.threshold_potential) == Ordering::Greater;

        //if triggered { new_potential = neuron.reset_potential};   document implementation
        if triggered { new_potential = testing_add(neuron.potential, -neuron.threshold_potential) }; //snn torch implementation

        (new_potential, triggered)
    };

    let faults_to_add = vec![
        (Fault::StuckAtZero, Unit::Multiplier),
        (Fault::StuckAtOne, Unit::Adder),
    ];

    let result = snn.test(1.0, &inputs, faults_to_add, 10, 2, f, expected_outputs).await;

    network::print_output(&result);
    renderer::render_to_html(&result, "./src/templates/template.hbs", "./src/templates/output.html");

}
