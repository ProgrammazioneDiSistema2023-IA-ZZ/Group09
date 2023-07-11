use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Mul};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use serde_json;
use rand::{Rng, thread_rng};

#[derive(Clone, Copy)]
enum FaultType {
    StuckAtZero(u8),
    StuckAtOne(u8),
    Transient(u8, i32),
}

impl Display for FaultType{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultType::StuckAtZero(bit) => {
                write!(f, "Bit {} stuck at zero", bit)
            }
            FaultType::StuckAtOne(bit) => {
                write!(f, "Bit {} stuck at one", bit)
            }
            FaultType::Transient(bit, time) => {
                write!(f, "Bit {} flipped at iteration {}", bit, time)
            }
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
enum OpSelector {
    FirstOperand,
    SecondOperand,
    Result,
}

#[derive(PartialEq, Clone)]
enum TestedUnit {
    //elaboration
    Adder(OpSelector, usize),
    Multiplier(OpSelector, usize),
    Comparator(OpSelector, usize),
    //memory
    SynapseWeight(usize, usize, bool),
    RestPotential(usize),
    ThresholdPotential(usize),
    ResetPotential(usize),
    Potential(usize),
    //communication
    NeuronInput(usize),
    NeuronOutput(usize),
}

impl Display for TestedUnit{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self{
            TestedUnit::Adder(op, neuron) => {
                write!(f, "Adder ({:?} affected) at neuron {}", op, neuron)
            }
            TestedUnit::Multiplier(op, neuron) => {
                write!(f, "Multiplier ({:?} affected) at neuron {}", op, neuron)
            }
            TestedUnit::Comparator(op, neuron) => {
                write!(f, "Comparator ({:?} affected) at neuron {}", op, neuron)
            }
            TestedUnit::SynapseWeight(neuron, edge, same_layer) => {
                write!(f, "Synapse weight {}° of {} layer from neuron {} ", edge, if *same_layer{"same"}else{"next"}, neuron)
            }
            TestedUnit::RestPotential(neuron) => {
                write!(f, "Rest Potential at neuron {}", neuron)
            }
            TestedUnit::ThresholdPotential(neuron) => {
                write!(f, "Threshold Potential at neuron {}", neuron)
            }
            TestedUnit::ResetPotential(neuron) => {
                write!(f, "Reset Potential at neuron {}", neuron)
            }
            TestedUnit::Potential(neuron) => {
                write!(f, "Potential at neuron {}", neuron)
            }
            TestedUnit::NeuronInput(neuron) => {
                write!(f, "Input of neuron {}", neuron)
            }
            TestedUnit::NeuronOutput(neuron) => {
                write!(f, "Output of neuron {}", neuron)
            }
        }
    }
}

//PUBLIC ENUM TO ADD VALUES

pub enum Fault {
    StuckAtZero,
    StuckAtOne,
    Transient,
}

pub enum Unit {
    //elaboration
    Adder,
    Multiplier,
    Comparator,
    //memory
    SynapseWeight,
    RestPotential,
    ThresholdPotential,
    ResetPotential,
    Potential,
    //communication
    NeuronInput,
    NeuronOutput,
}



#[derive(Serialize, Deserialize)]
pub struct SNN {
    layers: Vec<Layer>,
    neurons: Vec<Neuron>,
}

impl SNN {
    /// Create new empty Spiral Neural Network
    pub fn new() -> SNN {
        SNN {
            layers: Vec::<Layer>::new(),
            neurons: Vec::<Neuron>::new(),
        }
    }

    /// Create new Spiral Neural Network from JSON string
    /// # Arguments:
    /// * input: reference to String containing JSON object representing the SNN
    pub fn from_json(input : &String) -> SNN{
        serde_json::from_str(input).unwrap()
    }
    /// Create new layer in SNN
    pub fn new_layer(&mut self) -> usize {
        let ret = self.layers.len();
        self.layers.push(Layer {
            id: self.layers.len(),
            neurons: Vec::<usize>::new(),
        });
        ret
    }
    /// Create new neuron in SNN
    /// # Arguments
    /// * layer_id : layer identifier obtained from new_layer()
    /// * potential : initial potential of neuron
    /// * rest_potential : rest potential of the neuron
    /// * threshold_potential : threshold potential of the neuron
    /// * reset_potential : reset potential of the neuron
    /// * time_constant : time constant of the neuron membrane
    pub fn new_neuron(&mut self, layer_id: usize,
                  potential: f32,
                  rest_potential: f32,
                  threshold_potential: f32,
                  reset_potential: f32,
                  time_constant: f32) -> usize {
        let ret = self.neurons.len();
        self.neurons.push(Neuron {
            id: ret,
            layer: layer_id,
            potential,
            last_activity: 0,
            rest_potential,
            threshold_potential,
            reset_potential,
            time_constant,
            next_layer_synapses: Vec::<Synapses>::new(),
            same_layer_synapses: Vec::<Synapses>::new(),
        });
        self.layers[layer_id].add_neuron(ret);
        ret
    }
    async fn worker<'a>(neurons: Arc<RwLock<Vec<Neuron>>>, neuron_id: usize, input_signal: f32, current_step: i32, delta: f32, f: impl Fn(&Neuron, f32, i32, f32, Box<dyn Fn(f32, f32) -> f32 >, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> Ordering>) -> (f32, bool), fault: Option<Arc<(FaultType, TestedUnit)>>) -> Message {
        let neurons = neurons.read().await;
        let neuron = &neurons[neuron_id];

        let (new_potential, triggered) =
            match fault {
                None => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                Some(x) => {
                    let fault_type = (*x).0.clone();
                    let unit = &(*x).1;
                    match unit {
                        TestedUnit::Adder(selector, _) => {
                            match selector {
                                OpSelector::FirstOperand => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a, &fault_type.clone()).add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                OpSelector::SecondOperand => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(Self::add_fault_to_float(b, &fault_type.clone()))), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                OpSelector::Result => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a.add(b), &fault_type.clone())), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                            }
                        }
                        TestedUnit::Multiplier(selector, _) => {
                            match selector {
                                OpSelector::FirstOperand => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a, &fault_type.clone()).mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                OpSelector::SecondOperand => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(Self::add_fault_to_float(b, &fault_type.clone()))), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                OpSelector::Result => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a.mul(b), &fault_type.clone())), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                            }
                        }
                        TestedUnit::Comparator(selector, _) => {
                            match selector {
                                OpSelector::FirstOperand => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a, &fault_type.clone()).total_cmp(&b))),
                                OpSelector::SecondOperand => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&Self::add_fault_to_float(b, &fault_type.clone())))),
                                OpSelector::Result => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                            }
                        }
                        _ => f(neuron, input_signal, current_step, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b)))
                    }
                }
            };


        Message {
            neuron_id,
            new_potential,
            triggered,
        }
    }

    fn add_fault_to_float(value: f32, fault_type: &FaultType) -> f32 {
        match fault_type {
            FaultType::StuckAtZero(bit) => {
                let mask = 1 << bit;
                f32::from_bits(value.to_bits() & !mask)
            }
            FaultType::StuckAtOne(bit) => {
                let mask = 1 << bit;
                f32::from_bits(value.to_bits() | mask)
            }
            FaultType::Transient(bit, _) => {
                let mask = 1 << bit;
                if value.to_bits() & mask == 0 {
                    f32::from_bits(value.to_bits() | mask)
                } else {
                    f32::from_bits(value.to_bits() & !mask)
                }
            }
        }
    }


    ///
    /// Test the SNN over a given array of possible faults
    /// # Arguments
    ///
    /// * `delta`: time measure of a time step
    /// * `input_matrices`: vector of matrices of input signals
    /// * `faults_to_add`: vector of possible faults to be added to the test
    /// * `max_transient_iteration`: maximum time step when a fault of type "Transient" can be triggered
    /// * `n_inferences`: number of inferences of the input matrices each one randomly adding one of the faults in the fault_to_add array
    /// * `f`: activation function of the neuron
    ///
    /// # Activation Function
    ///
    /// The activation function accept:
    /// * neuron: &Neuron : reference to a Neuron
    /// * input_signal: f32 : the input signal of the neuron in the step (sum of the activated synapses' weights)
    /// * current_step : i32 : the current step of the execution
    /// * delta : f32 : the time measure of the time step
    /// * testing_add : Box<dyn Fn(f32, f32) -> f32> : adding function between f32 modified to accept faults
    /// * * example: a+b => a.testing_add(b)
    /// * testing_mul : Box<dyn Fn(f32, f32) -> f32> : multiplier function between f32 modified to accept faults
    /// * * example: a*b => a.testing_mul(b)
    /// * testing_cmp : Box<dyn Fn(f32, f32) -> Ordering> : comparator function between f32 modified to accept faults
    /// * * example: a == b => a.testing_cmp(b) == Ordering::Eq
    /// * * example: a > b => a.testing_cmp(b) == Ordering::Less
    /// * * example: a < b => a.testing_cmp(b) == Ordering::Greater
    ///
    /// returns pair of f32 as new calculated potential and bool as if the neuron was triggered
    ///
    /// Testing functions should be used instead of the +, * and <=> between f32 operations in order to allow testing of the adding, multiplier and comparator units.
    /// Anyways, if testing on those units is not required, the use of testing functions can be avoided.
    ///
    /// # Example
    ///
    /// Implementation of the Leaky Integrate and Fire model:
    ///
    /// ```
    ///    let f = |neuron: &Neuron, input_signal: f32, current_step: i32, delta: f32, testing_add: Box<dyn Fn(f32, f32) -> f32>, testing_mul: Box<dyn Fn(f32, f32) -> f32>, testing_cmp: Box<dyn Fn(f32, f32) -> Ordering>| {
    ///         let a = testing_add(neuron.potential, -neuron.rest_potential);
    ///         let b = testing_mul(testing_add(current_step as f32, -neuron.last_activity as f32), delta);
    ///         let c = (-b / neuron.time_constant).exp();
    ///         let mut new_potential = testing_add(testing_add(neuron.rest_potential, testing_mul(a, c)), input_signal);
    ///        let triggered = testing_cmp(new_potential, neuron.threshold_potential) == Ordering::Greater;
    ///         if triggered { new_potential = neuron.reset_potential };
    ///         (new_potential, triggered)
    ///     };
    /// ```
    pub async fn test(self, delta: f32, input_matrices: Vec<Vec<Vec<SignalInput>>>, faults_to_add: Vec<(Fault, Unit)>, max_transient_iteration: i32, n_inferences: usize, f: fn(&Neuron, f32, i32, f32, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> Ordering>) -> (f32, bool))  {

        let neurons = Arc::new(RwLock::new(self.neurons));

        let mut working_output = Vec::<Vec<i32>>::with_capacity(input_matrices.len());
        for input in &input_matrices{
            working_output.push(Self::run(neurons.clone(), &self.layers, delta, input, None, f).await);
        }


        let mut broken_outputs = Vec::<Vec<Vec<i32>>>::with_capacity(n_inferences);
        let mut faults_added = Vec::with_capacity(n_inferences);

        for i in 0..n_inferences{
            let neurons_read = neurons.read().await;
            let n_fault = thread_rng().gen_range(0..faults_to_add.len());
            let (ft, u) = &faults_to_add[n_fault];
            let broken_bit : u8 =  thread_rng().gen_range(0..32);
            let broken_neuron : usize =  thread_rng().gen_range(0..neurons_read.len().clone());
            let time : i32 = thread_rng().gen_range(0..max_transient_iteration);
            let op = match thread_rng().gen_range(0..3){
                0 => {OpSelector::FirstOperand}
                1 => {OpSelector::SecondOperand}
                _ => {OpSelector::Result}
            };
            let fault = match u{
                Unit::Adder => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::Adder(op, broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::Adder(op, broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::Adder(op, broken_neuron))}
                    }
                }
                Unit::Multiplier => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::Multiplier(op, broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::Multiplier(op, broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::Multiplier(op, broken_neuron))}
                    }
                }
                Unit::Comparator => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::Comparator(op, broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::Comparator(op, broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::Comparator(op, broken_neuron))}
                    }
                }
                Unit::SynapseWeight => {
                    let same_layer = thread_rng().gen_range(0..2);
                    let broken_edge= thread_rng().gen_range(0..if same_layer == 0{neurons_read[broken_neuron].same_layer_synapses.len()}else{neurons_read[broken_neuron].next_layer_synapses.len()});
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::SynapseWeight(broken_neuron, broken_edge, same_layer==0))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::SynapseWeight(broken_neuron, broken_edge, same_layer==0))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::SynapseWeight(broken_neuron, broken_edge, same_layer==0))}
                    }
                }
                Unit::RestPotential => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::RestPotential(broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::RestPotential(broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::RestPotential(broken_neuron))}
                    }
                }
                Unit::ThresholdPotential => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::ThresholdPotential(broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::ThresholdPotential(broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::ThresholdPotential(broken_neuron))}
                    }
                }
                Unit::ResetPotential => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::ResetPotential(broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::ResetPotential(broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::ResetPotential(broken_neuron))}
                    }
                }
                Unit::Potential => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::Potential(broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::Potential(broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::Potential(broken_neuron))}
                    }
                }
                Unit::NeuronInput => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::NeuronInput(broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::NeuronInput(broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::NeuronInput(broken_neuron))}
                    }
                }
                Unit::NeuronOutput => {
                    match ft {
                        Fault::StuckAtZero => {(FaultType::StuckAtZero(broken_bit), TestedUnit::NeuronOutput(broken_neuron))}
                        Fault::StuckAtOne => {(FaultType::StuckAtOne(broken_bit), TestedUnit::NeuronOutput(broken_neuron))}
                        Fault::Transient => {(FaultType::Transient(broken_bit, time), TestedUnit::NeuronOutput(broken_neuron))}
                    }
                }
            };
            let arc_fault = Arc::new(fault);
            drop(neurons_read);
            broken_outputs.push(vec![]);
            for input in &input_matrices{
                broken_outputs[i].push(Self::run(neurons.clone(), &self.layers, delta, input, Some(arc_fault.clone()), f).await);
                faults_added.push(arc_fault.clone());
            }
        }

        for i in 0..n_inferences{
            let res = Self::outputs_compare(&broken_outputs[i], &working_output);
            if res.len() != 0{
                println!("Found differences in test n°{}\n Fault: {}\n Unit: {}", i, faults_added[i].0, faults_added[i].1);
                for r in res{
                    println!("\t- input n°{} at output neuron {}", r.0, r.1);
                    println!("\t\twith fault:    {:?}", broken_outputs[i][r.0]);
                    println!("\t\twith no fault: {:?}", working_output[r.0]);
                }
            }
        }


    }

    fn outputs_compare(a : &Vec<Vec<i32>>, b : &Vec<Vec<i32>>) -> Vec<(usize, usize)> {
        let mut diff = Vec::<(usize, usize)>::new();
        for i in 0..a.len(){
            for j in 0..a[i].len(){
                if a[i][j]!=b[i][j]{
                    diff.push((i,j));
                }
            }
        }
        diff
    }


    async fn run(neurons: Arc<RwLock<Vec<Neuron>>>, layers: &Vec<Layer>, delta: f32, input_matrix: &Vec<Vec<SignalInput>>, fault: Option<Arc<(FaultType, TestedUnit)>>, f: fn(&Neuron, f32, i32, f32, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> Ordering>) -> (f32, bool)) -> Vec<i32> {
        let mut neurons_write = neurons.write().await;

        neurons_write.iter_mut().for_each(|n| n.initialize());
        let mut original_value: f32 = f32::NAN;


        if let Some(x) = fault.clone() {
            let (ft, u) = &*x;
            match u {
                TestedUnit::SynapseWeight(neuron, edge, same_layer) => {
                    if *same_layer {
                        original_value = neurons_write[*neuron].same_layer_synapses[*edge].weight;
                        match ft {
                            FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                neurons_write[*neuron].same_layer_synapses[*edge].weight =
                                    Self::add_fault_to_float(original_value, ft);
                            }
                            _ => {}
                        }
                    } else {
                        original_value = neurons_write[*neuron].next_layer_synapses[*edge].weight;
                        match ft {
                            FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                neurons_write[*neuron].next_layer_synapses[*edge].weight =
                                    Self::add_fault_to_float(original_value, ft);
                            }
                            _ => {}
                        }
                    }
                }
                TestedUnit::RestPotential(neuron) => {
                    original_value = neurons_write[*neuron].rest_potential;
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            neurons_write[*neuron].rest_potential = Self::add_fault_to_float(original_value, ft);
                        }
                        _ => {}
                    }
                }
                TestedUnit::ThresholdPotential(neuron) => {
                    original_value = neurons_write[*neuron].threshold_potential;
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            neurons_write[*neuron].threshold_potential = Self::add_fault_to_float(original_value, ft);
                        }
                        _ => {}
                    }
                }
                TestedUnit::ResetPotential(neuron) => {
                    original_value = neurons_write[*neuron].reset_potential;
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            neurons_write[*neuron].reset_potential = Self::add_fault_to_float(original_value, ft);
                        }
                        _ => {}
                    }
                }
                TestedUnit::Potential(neuron) => {
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            neurons_write[*neuron].potential = Self::add_fault_to_float(neurons_write[*neuron].potential, ft);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let last_layer_first_neuron_id = layers[layers.len() - 1].neurons[0];
        let mut ret = vec![0; layers.last().unwrap().neurons.len()];


        let mut t = 0;
        let mut results = SignalOutput::new_initialized_vec(neurons_write.len());
        let mut input_iter = input_matrix.iter();
        let mut n_to_trigger = 0;
        let mut countdown = layers.len();

        drop(neurons_write);

        /*        let broken_f = match fault {
                    None => {None}
                    Some((ft, u)) => {
                        if u == TestedUnit::Adder || u == TestedUnit::Multiplier || u == TestedUnit::Comparator{
                            Some(Self::bind_f_with_fault(f, ft, u))
                        }
                        else {None}
                    }
                };
                let f = Self::bind_f(f);

                let fault = Arc::new(fault);

                let g = Arc::new(Self::bind_to_f(f, fault));*/


        loop {
            t += 1;
            match input_iter.next() {
                None => {
                    if countdown == 0 {
                        break;
                    }
                    countdown -= 1;
                }
                Some(signals) => {
                    for signal in signals {
                        results[signal.neuron_id].weight += signal.weight;
                        results[signal.neuron_id].to_trigger = true;
                    }
                    n_to_trigger += signals.len();
                }
            }

            let mut handles = Vec::with_capacity(n_to_trigger);

            for i in 0..results.len() {
                if results[i].to_trigger {
                    let cloned_neurons = neurons.clone();
                    let mut cloned_weight = results[i].weight.clone();
                    let mut cloned_fault = None;

                    //ADDER MULT CMP AND NEURON INPUT TEST
                    if let Some(x) = fault.clone() {
                        let (ft, u) = &*x;
                        if let TestedUnit::NeuronInput(n) = u {
                            if i == *n {
                                match ft {
                                    FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                        cloned_weight = Self::add_fault_to_float(results[i].weight.clone(), ft);
                                    }
                                    FaultType::Transient(_, time) => {
                                        if *time == t {
                                            cloned_weight = Self::add_fault_to_float(results[i].weight.clone(), ft)
                                        }
                                    }
                                }
                            }
                        }
                        if let TestedUnit::Adder(_, n) | TestedUnit::Multiplier(_, n) | TestedUnit::Comparator(_, n) = u {
                            if i == *n {
                                match ft {
                                    FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                        cloned_fault = fault.clone();
                                    }
                                    FaultType::Transient(_, time) => {
                                        if *time == t {
                                            cloned_fault = fault.clone();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    handles.push(
                        tokio::spawn(async move {
                            Self::worker(cloned_neurons, i, cloned_weight, t, delta, f, cloned_fault)
                        })
                    )
                }
                results[i].to_trigger = false;
                results[i].weight = 0.0;
            }
            if handles.len() == 0 { break; }
            let mut tokyo_results = Vec::with_capacity(handles.len());
            for handle in handles {
                tokyo_results.push(handle.await.unwrap().await);
            }

            let mut neuron_found = false;
            let mut neurons_write = neurons.write().await;
            for message in tokyo_results {
                neurons_write[message.neuron_id].potential = message.new_potential;
                neurons_write[message.neuron_id].last_activity = t;

                if message.triggered {
                    //NEURON OUTPUT TEST
                    if let Some(x) = fault.clone() {
                        let (ft, u) = &*x;
                        if let TestedUnit::NeuronOutput(n) = u {
                            if *n == message.neuron_id {
                                match ft {
                                    FaultType::StuckAtZero(_) => {
                                        continue;
                                        //acting like the neuron was not triggered
                                    }
                                    FaultType::StuckAtOne(_) => {
                                        neuron_found = true;
                                        //the neuron was triggered, so later there's no need to act like it was triggered
                                    }
                                    FaultType::Transient(_, time) => {
                                        if *time == t {
                                            neuron_found = true;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for synapse in neurons_write[message.neuron_id].next_layer_synapses.iter() {
                        results[synapse.to].to_trigger = true;
                        results[synapse.to].weight += synapse.weight; //NEURON OUTPUT
                    }
                    for synapse in neurons_write[message.neuron_id].same_layer_synapses.iter() {
                        results[synapse.to].to_trigger = true;
                        results[synapse.to].weight += synapse.weight;   //NEURON OUTPUT
                    }
                    if message.neuron_id >= last_layer_first_neuron_id {
                        ret[message.neuron_id - last_layer_first_neuron_id] += 1;
                    }
                }
            }
            //NEURON OUTPUT, POTENTIAL, AND STATIC VALUE IN TRANSIENT MODE TEST
            if let Some(x) = fault.clone() {
                let (ft, u) = &*x;
                match u {
                    TestedUnit::SynapseWeight(neuron, edge, same_layer) => {
                        if let FaultType::Transient(_, time) = ft {
                            if *time == t {
                                if *same_layer {
                                    neurons_write[*neuron].same_layer_synapses[*edge].weight =
                                        Self::add_fault_to_float(original_value, ft);
                                } else {
                                    neurons_write[*neuron].next_layer_synapses[*edge].weight =
                                        Self::add_fault_to_float(original_value, ft);
                                }
                            }
                        }
                    }
                    TestedUnit::RestPotential(n) => {
                        if let FaultType::Transient(_, time) = ft {
                            if *time == t {
                                neurons_write[*n].rest_potential = Self::add_fault_to_float(original_value, ft);
                            }
                        }
                    }
                    TestedUnit::ThresholdPotential(n) => {
                        if let FaultType::Transient(_, time) = ft {
                            if *time == t {
                                neurons_write[*n].threshold_potential = Self::add_fault_to_float(original_value, ft);
                            }
                        }
                    }
                    TestedUnit::ResetPotential(n) => {
                        if let FaultType::Transient(_, time) = ft {
                            if *time == t {
                                neurons_write[*n].reset_potential = Self::add_fault_to_float(original_value, ft);
                            }
                        }
                    }
                    TestedUnit::Potential(n) => {
                        match ft {
                            FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                neurons_write[*n].potential = Self::add_fault_to_float(neurons_write[*n].potential, ft);
                            }
                            FaultType::Transient(_, time) => {
                                if t == *time {
                                    neurons_write[*n].potential = Self::add_fault_to_float(neurons_write[*n].potential, ft);
                                }
                            }
                        }
                    }
                    TestedUnit::NeuronOutput(n) => {
                        match ft {
                            FaultType::StuckAtZero(_) => {}
                            FaultType::StuckAtOne(_) => {
                                if !neuron_found {
                                    for synapse in neurons_write[*n].next_layer_synapses.iter() {
                                        results[synapse.to].to_trigger = true;
                                        results[synapse.to].weight += synapse.weight; //NEURON OUTPUT
                                    }
                                    for synapse in neurons_write[*n].same_layer_synapses.iter() {
                                        results[synapse.to].to_trigger = true;
                                        results[synapse.to].weight += synapse.weight;   //NEURON OUTPUT
                                    }
                                    if *n >= last_layer_first_neuron_id {
                                        ret[*n - last_layer_first_neuron_id] += 1;
                                    }
                                }
                            }
                            FaultType::Transient(_, time) => {
                                if t == *time && !neuron_found {
                                    for synapse in neurons_write[*n].next_layer_synapses.iter() {
                                        results[synapse.to].to_trigger = true;
                                        results[synapse.to].weight += synapse.weight; //NEURON OUTPUT
                                    }
                                    for synapse in neurons_write[*n].same_layer_synapses.iter() {
                                        results[synapse.to].to_trigger = true;
                                        results[synapse.to].weight += synapse.weight;   //NEURON OUTPUT
                                    }
                                    if *n >= last_layer_first_neuron_id {
                                        ret[*n - last_layer_first_neuron_id] += 1;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            drop(neurons_write);
        }


        //CLEANUP
        if let Some(x) = fault.clone() {
            let (_, u) = &*x;
            let mut neurons_write = neurons.write().await;
            match u {
                TestedUnit::SynapseWeight(neuron, edge, same_layer) => {
                    if *same_layer {
                        neurons_write[*neuron].same_layer_synapses[*edge].weight = original_value;
                    } else {
                        neurons_write[*neuron].next_layer_synapses[*edge].weight = original_value;
                    }
                }
                TestedUnit::RestPotential(neuron) => {
                    neurons_write[*neuron].rest_potential = original_value;
                }
                TestedUnit::ThresholdPotential(neuron) => {
                    neurons_write[*neuron].threshold_potential = original_value;
                }
                TestedUnit::ResetPotential(neuron) => {
                    neurons_write[*neuron].reset_potential = original_value;
                }
                TestedUnit::Potential(neuron) => {
                    neurons_write[*neuron].potential = original_value;
                }
                _ => {}
            }
        }
        ret
    }
}

#[derive(Serialize, Deserialize)]
struct Layer {
    id: usize,
    neurons: Vec<usize>,
}

impl Layer {
    fn add_neuron(&mut self, neuron_id: usize) {
        self.neurons.push(neuron_id);
    }
}

/// Neuron parameters
///
/// * potential: neuron potential value
/// * threshold_potential: neuron threshold potential value
/// * rest_potential: neuron rest potential value
/// * reset_potential: neuron reset potential value
/// * time_constant: neuron time constant value
/// * last_activity: last step when  the neuron potential was recalculated
///   (in the activation function this value can be used with the current_step to calculate the steps passed)
///
///
#[derive(Serialize, Deserialize)]
pub struct Neuron {
    id: usize,
    layer: usize,
    pub potential: f32,
    pub last_activity: i32,
    pub rest_potential: f32,
    pub threshold_potential: f32,
    pub reset_potential: f32,
    pub time_constant: f32,
    next_layer_synapses: Vec<Synapses>,
    same_layer_synapses: Vec<Synapses>,
}

impl Neuron {
    pub fn add_same_layer_synapse(&mut self, neuron: usize, weight: f32) {
        self.same_layer_synapses.push(Synapses {
            to: neuron,
            weight,
        });
    }
    pub fn add_next_layer_synapse(&mut self, neuron: usize, weight: f32) {
        self.next_layer_synapses.push(Synapses {
            to: neuron,
            weight,
        });
    }

    fn initialize(&mut self) {
        self.last_activity = 0;
        self.potential = self.reset_potential;
    }
}

#[derive(Serialize, Deserialize)]
struct Synapses {
    to: usize,
    weight: f32,
}

struct Message {
    neuron_id: usize,
    new_potential: f32,
    triggered: bool,
}

#[derive(Serialize, Deserialize)]
pub struct SignalInput {
    neuron_id: usize,
    weight: f32,
}

/// Create new input for the Spiral Neural Network from JSON string
/// # Arguments:
/// * input: reference to String containing JSON object representing the input
impl SignalInput{
    pub fn from_json(input : &String)->Vec<Vec<SignalInput>>{
        serde_json::from_str(input).unwrap()
    }
}

struct SignalOutput {
    to_trigger: bool,
    weight: f32,
}

impl SignalOutput {
    fn new_initialized_vec(size: usize) -> Vec<Self> {
        let mut ret = Vec::<Self>::with_capacity(size);
        for _i in 0..size {
            ret.push({
                Self {
                    to_trigger: false,
                    weight: 0.0,
                }
            })
        }
        ret
    }
}


fn main() {}