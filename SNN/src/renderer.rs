use std::fs;
use handlebars::{Handlebars};
use crate::network::Output;
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
struct OutputWithSumsJson{
    expected_output: u8,
    no_fault_sum: Vec<i32>,
    faulted_sum: Vec<OutputFaultedWithSumsJson>,
    signals: Vec<OutputJson>
}

#[derive(Serialize, Deserialize)]
struct OutputFaultedWithSumsJson{
    fault: String,
    values: Vec<i32>
}

#[derive(Serialize, Deserialize)]
struct OutputJson {
    no_fault: Vec<bool>,
    faulted: Vec<OutputFaultedJson>,
}

#[derive(Serialize, Deserialize)]
struct OutputFaultedJson {
    different: bool,
    fault: String,
    values: Vec<bool>,
}

fn to_json(output: &Output) -> OutputWithSumsJson {
    let mut ret = OutputWithSumsJson{
        expected_output: output.expected_output.value,
        no_fault_sum: vec![0; output.no_fault_output[0].len()],
        faulted_sum: vec![],
        signals: vec![],
    };
    for nf in output.no_fault_output.iter(){
        for (bb,b) in nf.iter().enumerate(){
            if *b {
                ret.no_fault_sum[bb]+=1;
            }
        }
    }
    for wf_outer in output.with_fault_output.iter(){
        let mut tmp = OutputFaultedWithSumsJson{ fault:  format!("{} at unit: {}", wf_outer.fault, wf_outer.unit).to_string(), values: vec![0; output.no_fault_output[0].len()] };
        for wf in wf_outer.output.iter(){
            for (bb,b) in wf.iter().enumerate(){
                if *b {
                    tmp.values[bb]+=1;
                }
            }
        }
        ret.faulted_sum.push(tmp);
    }
    for i in 0..output.no_fault_output.len() {
        let no_fault = output.no_fault_output[i].clone();
        let mut faulted = vec![];
        for wf in output.with_fault_output.iter() {
            let mut different = false;
            for j in 0..no_fault.len() {
                if wf.output[i][j] != no_fault[j] { different = true; }
            }
            faulted.push(
                OutputFaultedJson { different, fault: format!("{} at unit: {}", wf.fault, wf.unit).to_string(), values: wf.output[i].clone() }
            );
        }
        ret.signals.push(OutputJson { no_fault, faulted });
    }
    ret
}

/// Create a new HTML file to visualize data
/// # Arguments:
/// * data : vector of outputs to display
/// * template_path : template file's path where insert data
/// * output_path : output HTML file path
pub fn render_to_html(data: &Vec<Output>, template_path: &str, output_path: &str){
    let mut reg = Handlebars::new();
    reg.register_template_file("file", template_path).expect("Unable to read template file");

    let data: Vec<OutputWithSumsJson> = data.iter().map(|x|to_json(x)).collect();
    match reg.render("file", &data){
        Ok(x) => {
            fs::write(output_path, x).expect("Unable to write html file");
        }
        Err(x) => {println!("{}", x);}
    }
}
