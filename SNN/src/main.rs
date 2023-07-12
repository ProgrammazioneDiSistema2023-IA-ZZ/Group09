use std::cmp::Ordering;
use std::fs;
mod lib;
use lib::{Fault, Neuron, SNN, Unit, SignalInput};
use crate::lib::print_output;

#[tokio::main]
async fn main() {
    //let json_string = fs::read_to_string("snn.json").unwrap();
    let json_string = fs::read_to_string("C:\\Users\\Rosso\\Documents\\Universita'\\Programmazione di sistema\\pds_project\\Group09\\SNN\\src\\snn.json").unwrap();
    let snn: SNN = SNN::from_json(&json_string);


    //let json_string = fs::read_to_string("input.json").unwrap();
    let json_string = fs::read_to_string("C:\\Users\\Rosso\\Documents\\Universita'\\Programmazione di sistema\\pds_project\\Group09\\SNN\\src\\input.json").unwrap();
    let input= SignalInput::from_json(&json_string);
    //let json_string = fs::read_to_string("input2.json").unwrap();
    let json_string = fs::read_to_string("C:\\Users\\Rosso\\Documents\\Universita'\\Programmazione di sistema\\pds_project\\Group09\\SNN\\src\\input2.json").unwrap();
    let input2= SignalInput::from_json(&json_string);


    let f = |neuron: &Neuron, input_signal: f32, current_step: i32, delta: f32, testing_add: Box<dyn Fn(f32, f32) -> f32>, testing_mul: Box<dyn Fn(f32, f32) -> f32>, testing_cmp: Box<dyn Fn(f32, f32) -> Ordering>| {
        let a = testing_add(neuron.potential, -neuron.rest_potential);
        let b = testing_mul(testing_add(current_step as f32, -neuron.last_activity as f32), delta);
        let c = (-b / neuron.time_constant).exp();
        let mut new_potential = testing_add(testing_add(neuron.rest_potential, testing_mul(a, c)), input_signal);
        let triggered = testing_cmp(new_potential, neuron.threshold_potential) == Ordering::Greater;
        if triggered { new_potential = neuron.reset_potential };
        (new_potential, triggered)
    };

    let inputs = &vec![input, input2];

    let ret = snn.test(1.0, &inputs, vec![(Fault::Transient, Unit::ThresholdPotential),(Fault::StuckAtZero, Unit::NeuronInput)],10, 10, f).await;
    print_output(&ret);

}
