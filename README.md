# Spiking Neural Networks and Resilience

Testing of resilience of spiking neural network hardware implementation. 

## Build and Run

You will need a Rust development environment, which you can install by visiting https://rustup.rs/ and following the instructions.

To build the project use:

```
cargo build
```

To run the example main, use:

```
cargo run
```

Following the instructions on how the project is structured and on how to simulate the inferences with the faults.

## Structure of the project

The spiking neural network is described by the SNN struct, containing:
- `layers` : vector of layer identifier of the snn
- `neurons:`  vector of neuron identifier of the snn

The SNN struct implement the functions:
- `new` : creates a new empty SNN
- `from_json` : accept a reference to String of the json dump of the SNN, return the serialized struct SNN
- `new_layer` : creates a new layer returning the layer identifier
- `new_neuron` : creates a new neuron, accepting the layer identifier, the potential, the rest potential, the reset potential, the threshold potential and the time constant, and returns the neuron identifier.
- `new_synapse` : that accept the two neuron identifiers and the weight of the synapse, add a synapse



Layers are described by the Layer struct, with :
- `id` : layer identifier
- `neurons` : vector of neuron identifier contained in the layer

Neurons are described by the Neuron struct, with:
- `id` : neuron identidier
- `layer_id` : layer identidier
- `potential` : neuron potential value
- `threshold_potential` : neuron threshold potential value
- `rest_potential` : neuron rest potential value
- `reset_potential` : neuron reset potential value
- `time_constant` : neuron time constant value
- `last_activity` : last step when  the neuron potential was recalculated
  (in the activation function this value can be used with the current_step to calculate the steps passed)
- `next_layer_synapses` : vector of synapses in the same layer
- `next_layer_synapses` : vector of synapses in the next layer

## Faults

Dfferent kind of faults can be added to the function in order to test the resiliance of the SNN.

There are three types of faults that can be selected from the Fault enum. Here are the values:

- `StuckAtZero` : one random bit of the unit is set to zero for the duration of the inference
- `StuckAtOne` : one random bit of the unit is set to one for the duration of the inference
- `Transient` : one random bit of the unit is flipped in a random moment during the inference

There are different unit to add faults to that can be selected from the Unit enum. Here are the values:

*elaboration units*

- `Adder` : adder component working in the neuron activation function. The fault is randomically added to the result, to the first addend or to the second addend.

- `Multiplier` : multiplier component working in the neuron activation function. The fault is randomically added to the result, to the first factor or to the second factor.

- `Comparator` : comparator component working in the neuron activation function. The fault is randomically added to the left hand side or the right hand side of the comparison.

*memory units*

- `SynapseWeight` : weight of a synapse connecting two neurons. The fault is added to a randomically chosen synapse in the SNN.

- `RestPotential` : rest potential of a neuron. The fault is added to a randomically chosen neuron in the SNN.

- `ThresholdPotential` : threshold potential of a neuron. The fault is added to a randomically chosen neuron in the SNN.
- `ResetPotential` : reset potential of a neuron. The fault is added to a randomically chosen neuron in the SNN.
- `Potential` : potential of a neuron. The fault is added to a randomically chosen neuron in the SNN.

*communication*

- `NeuronInput` : input value of a neuron after each iteration (sum of all weights of the synapses connected to a triggered neuron). The fault is added to the input value of a randomically chosen neuron in the SNN.

- `NeuronOutput` : output value of a neuron after each iteration (is the neuron triggered or not). The fault is added to the output value of a randomically chosen neuron in the SNN. In this case the stuck at zero fault force the neuron to never trigger, the stuck at one to always trigger, and the transient to trigger when it's not and viceversa in one random moment during the inference.


## Test function
Test the SNN over a given array of possible faults.

### Arguments

- `delta: f32` : time measure of a time step
- `input_matrices: Vec<Vec<SignalInput>>` : vector of matrices of input signals [specifications](#input)
- `faults_to_add: Vec<(Fault, Unit)>` : vector of possible faults to be added to the test
- `max_transient_iteration: i32` : maximum time step when a fault of type "Transient" can be triggered
- `n_inferences: usize` : number of inferences of the input matrices each one randomly adding one of the faults in the fault_to_add array
- `f` : activation function of the neuron [specifications](#activation-function)

### Return

Vector of Outputs, containing for each input the output with no fault added and the list of outputs with the faults added [specifications](#output)

### Activation Function

The activation function accept:
- `neuron: &Neuron `: reference to a Neuron
- `input_signal: f32 `: the input signal of the neuron in the step (sum of the activated synapses' weights)
- `current_step : i32 `: the current step of the execution
- `delta : f32` : the time measure of the time step

- `testing_add : Box<dyn Fn(f32, f32) -> f32>` : adding function between f32 modified to accept faults

- `testing_mul : Box<dyn Fn(f32, f32) -> f32>` : multiplier function between f32 modified to accept faults

- `testing_cmp : Box<dyn Fn(f32, f32) -> Ordering>` : comparator function 

Returns a pair (f32, bool) as new calculated potential and whether the neuron was triggered or not.

Testing functions should be used instead of the +, * and <=> between f32 operations in order to allow testing of the adding, multiplier and comparator units, like in the example below:

```
// a+b
let result = a.testing_add(b);

// a*b
let result = a.testing_mul(b);

// a == b 
let result = a.testing_cmp(b) == Ordering::Eq;
// a > b
let result = a.testing_cmp(b) == Ordering::Less;
// a < b
let result = a.testing_cmp(b) == Ordering::Greater;
```


However, if testing on those units is not specified in the *faults_to_add* parameter of the *test* function, the use of testing functions can be avoided.

#### Example

Implementation of the Leaky Integrate and Fire model:

```
let f = |neuron: &Neuron, input_signal: f32, current_step: i32, delta: f32, 
            testing_add: Box<dyn Fn(f32, f32) -> f32>, 
            testing_mul: Box<dyn Fn(f32, f32) -> f32>, 
            testing_cmp: Box<dyn Fn(f32, f32) -> Ordering>| {
        
    let a = testing_add(neuron.potential, -neuron.rest_potential);

    let b = testing_mul(testing_add(current_step as f32, -neuron.last_activity as f32), delta);

    let c = (-b / neuron.time_constant).exp();

    let mut new_potential = testing_add(testing_add(neuron.rest_potential, testing_mul(a, c)), input_signal);

    let triggered = testing_cmp(new_potential, neuron.threshold_potential) == Ordering::Greater;

    if triggered { new_potential = neuron.reset_potential };

    (new_potential, triggered)
};
 ```

## Input


The *test* function accepts a vector of input matrices of type Vec<Vec\<SignalInput>>.

The SignalInput struct is composed by the neuron identifier (belonging to the first layer) and the weight of the input.

It implement one function called *from_json* that accept a reference to string of the json rapresentation of the matrix of SignalInput and returns the actual matrix.

## Output

The *test* function returns a vector of outputs each one having:

- input: reference to input matrix
- no_fault_output: output matrix with no fault added
- with_fault_added: vector of outputs with fault added

Each output with fault added is composed by:

- output: output matrix obtained after the fault were added
- fault: type of fault added
- unit: unit whom fault was added

### Print output

The result of the test function can be pretty printend using the function *print_output*.

An example can be: 

 ```

Input #0

#################

NO FAULT OUTPUT:

[0 0 0] [0 0 0] [0 1 0] [0 0 0] [0 1 0] [0 1 0] [0 0 0] [0 1 0] [0 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [0 0 0] 

#################

FAULTED OUTPUTS:

adding fault: Bit 20 flipped at iteration 4 at unit: Threshold Potential at neuron 9: 
[0 0 0] [0 0 0] [0 1 0] [0 0 0] [0 1 0] [0 1 0] [0 0 0] [0 1 0] [0 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [0 0 0] 

adding fault: Bit 0 flipped at iteration 6 at unit: Threshold Potential at neuron 2: 
[0 0 0] [0 0 0] [0 1 0] [0 0 0] [0 1 0] [0 1 0] [0 0 0] [0 1 0] [0 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [0 0 0] 


Input #1

#################

NO FAULT OUTPUT:

[0 0 0] [0 0 0] [0 1 0] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [0 0 0] 

#################

FAULTED OUTPUTS:

adding fault: Bit 20 flipped at iteration 4 at unit: Threshold Potential at neuron 9: 
[0 0 0] [0 0 0] [0 1 0] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [0 0 0] 

adding fault: Bit 0 flipped at iteration 6 at unit: Threshold Potential at neuron 2: 
[0 0 0] [0 0 0] [0 1 0] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [1 1 1] [0 0 0] 


 ```