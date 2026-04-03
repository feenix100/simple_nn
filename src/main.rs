// This function is used to generate the activation for each neuron.
// It takes any f64 number and squeezes it into a value between 0.0 and 1.0.
fn sigmoid(x: f64) -> f64 {
    // 1 / (1 + e^-x)
    1.0 / (1.0 + (-x).exp())
}

// This is the derivative of the sigmoid function.
// We use it during training to figure out how much to adjust weights.
fn sigmoid_derivative(output: f64) -> f64 {
    // When we already know the sigmoid output, its derivative is output * (1 - output).
    output * (1.0 - output)
}

// This struct stores the entire neural network.
struct NeuralNet {
    // w1 holds the weights from the 6 input values to the 4 hidden neurons.
    // That means 4 rows (one per hidden neuron), each with 6 input weights.
    w1: [[f64; 6]; 4],

    // b1 holds one bias value for each hidden neuron.
    b1: [f64; 4],

    // w2 holds the weights from the 4 hidden neurons to the final output neuron.
    w2: [f64; 4],

    // b2 is the bias for the final output neuron.
    b2: f64,
}

// This impl block defines methods that belong to NeuralNet.
impl NeuralNet {
    // Create a new neural network with starting weights and biases.
    fn new() -> Self {
        // Return a NeuralNet instance.
        Self {
            // Initial weights from input layer -> hidden layer.
            w1: [
                // Hidden neuron 0 weights for the 6 input features.
                [0.20, 0.30, 0.25, 0.10, 0.10, -0.20],
                // Hidden neuron 1 weights.
                [0.15, 0.25, 0.20, 0.30, 0.10, -0.10],
                // Hidden neuron 2 weights.
                [0.25, 0.20, 0.30, 0.10, 0.30, -0.15],
                // Hidden neuron 3 weights.
                [0.10, 0.15, 0.20, 0.25, 0.20, -0.25],
            ],

            // Start all hidden biases at 0.0.
            b1: [0.0; 4],

            // Initial weights from hidden layer -> output neuron.
            w2: [0.20, 0.25, 0.30, 0.15],

            // Start output bias at 0.0.
            b2: 0.0,
        }
    }

    // Run one forward pass through the network.
    // Input: [f64; 6] because we have 6 input features.
    // Output: ([f64; 4], f64) meaning:
    //   - the 4 hidden neuron outputs
    //   - the final output score
    fn forward(&self, input: [f64; 6]) -> ([f64; 4], f64) {
        // Create an array to store the 4 hidden neuron outputs.
        let mut hidden = [0.0; 4];

        // Loop over each hidden neuron index h = 0..4.
        for h in 0..4 {
            // Start the neuron's sum with its bias.
            let mut sum = self.b1[h];

            // Loop over each of the 6 input features.
            for i in 0..6 {
                // Multiply input[i] by its weight into hidden neuron h and add it to the sum.
                sum += self.w1[h][i] * input[i];
            }

            // Apply sigmoid activation so the hidden neuron output is between 0 and 1.
            hidden[h] = sigmoid(sum);
        }

        // Start the output neuron's sum with the output bias.
        let mut output_sum = self.b2;

        // Add contribution from each hidden neuron.
        for h in 0..4 {
            // hidden[h] * output weight for that hidden neuron
            output_sum += self.w2[h] * hidden[h];
        }

        // Apply sigmoid to get the final output score between 0 and 1.
        let output = sigmoid(output_sum);

        // Return both the hidden layer outputs and the final output.
        (hidden, output)
    }

    // Train the network using a slice of training examples.
    // data contains pairs of:
    //   ([f64; 6], f64)
    // which means:
    //   (input_features, target_score)
    fn train(&mut self, data: &[([f64; 6], f64)], epochs: usize, learning_rate: f64) {
        // Print a decorative top border before training starts.
        println!("\n{}", line('═', 64));
        // Print a centered training title.
        println!("{:^64}", "TRAINING NEURAL NETWORK");
        // Print another border line.
        println!("{}", line('═', 64));

        // Loop for the requested number of training passes.
        for epoch in 0..epochs {
            // Track total loss for this epoch.
            let mut total_loss = 0.0;

            // Loop through every training example.
            for &(input, target) in data {
                // Run the current input through the network.
                let (hidden, output) = self.forward(input);

                // Calculate how far off the prediction was.
                let error = target - output;

                // Add squared error to total loss.
                total_loss += error * error;

                // Compute output delta for backpropagation.
                // This says how much the output neuron should change.
                let output_delta = error * sigmoid_derivative(output);

                // Create an array to hold delta values for each hidden neuron.
                let mut hidden_deltas = [0.0; 4];

                // Compute delta for each hidden neuron.
                for h in 0..4 {
                    // Hidden delta depends on:
                    // - output layer weight from hidden neuron h
                    // - output delta
                    // - derivative of hidden neuron activation
                    hidden_deltas[h] =
                        self.w2[h] * output_delta * sigmoid_derivative(hidden[h]);
                }

                // Update hidden -> output weights.
                for h in 0..4 {
                    // Standard gradient-descent-style update.
                    self.w2[h] += learning_rate * output_delta * hidden[h];
                }

                // Update output bias.
                self.b2 += learning_rate * output_delta;

                // Update input -> hidden weights.
                for h in 0..4 {
                    // Loop across the 6 input features.
                    for i in 0..6 {
                        // Adjust the input->hidden weight based on hidden delta and input value.
                        self.w1[h][i] += learning_rate * hidden_deltas[h] * input[i];
                    }

                    // Update the hidden bias for neuron h.
                    self.b1[h] += learning_rate * hidden_deltas[h];
                }
            }

            // Every 1000 epochs, print progress.
            if epoch % 1000 == 0 {
                // Show epoch number and average loss.
                println!(
                    "Epoch {:>5}   |   Avg Loss: {:.6}",
                    epoch,
                    total_loss / data.len() as f64
                );
            }
        }

        // Print a divider after training finishes.
        println!("{}", line('─', 64));
        // Print a centered completion message.
        println!("{:^64}", "Training complete");
        // Print bottom border.
        println!("{}", line('═', 64));
    }

    // Make a prediction from a feature vector.
    fn predict(&self, input: [f64; 6]) -> f64 {
        // We only care about the final output, not the hidden layer values.
        let (_, output) = self.forward(input);
        // Return the final score.
        output
    }
}

// Helper function to build one 6-feature input vector.
// The feature order is:
// [has_link, asks_sensitive_info, urgent_language, mentions_package, mentions_job, trusted_sender]
fn msg(
    has_link: f64,
    asks_sensitive_info: f64,
    urgent_language: f64,
    mentions_package: f64,
    mentions_job: f64,
    trusted_sender: f64,
) -> [f64; 6] {
    // Return the values as an array in the exact order the neural network expects.
    [
        has_link,
        asks_sensitive_info,
        urgent_language,
        mentions_package,
        mentions_job,
        trusted_sender,
    ]
}

// Build a line made of the same character repeated len times.
// Example: line('═', 10) returns "══════════"
fn line(ch: char, len: usize) -> String {
    // Create an iterator of repeated characters,
    std::iter::repeat(ch)
        // limit it to len characters,
        .take(len)
        // and collect it into a String.
        .collect()
}

// Build a text progress bar for the scam score.
fn score_bar(score: f64, width: usize) -> String {
    // Figure out how many blocks should be filled based on the score.
    let filled = (score * width as f64).round() as usize;

    // Create an empty String to hold the bar.
    let mut bar = String::new();

    // Loop across every position in the bar.
    for i in 0..width {
        // If this position is inside the filled region,
        if i < filled {
            // add a solid block.
            bar.push('█');
        } else {
            // Otherwise add an empty block.
            bar.push('░');
        }
    }

    // Return the finished bar string.
    bar
}

// Convert the numeric score into a human-readable label.
fn prediction_label(score: f64) -> &'static str {
    // If score is at least 0.5, call it likely scam.
    if score >= 0.5 {
        "LIKELY SCAM"
    } else {
        // Otherwise call it likely safe.
        "LIKELY SAFE"
    }
}

// Print the intro section shown when the program starts.
fn print_intro() {
    // Top border.
    println!("{}", line('═', 64));
    // Title line.
    println!("{:^64}", "SCAM TEXT NEURAL NETWORK");
    // Subtitle line.
    println!("{:^64}", "Simple Rust Demo");
    // Another border.
    println!("{}", line('═', 64));
    // Describe the feature list.
    println!("Features used by the model:");
    println!("  [1] has_link");
    println!("  [2] asks_sensitive_info");
    println!("  [3] urgent_language");
    println!("  [4] mentions_package");
    println!("  [5] mentions_job");
    println!("  [6] trusted_sender");
    // Divider line.
    println!("{}", line('─', 64));
}

// Print one full prediction block for one test message.
fn print_prediction(message: &str, input: [f64; 6], score: f64) {
    // Blank line for spacing.
    println!();
    // Header border.
    println!("{}", line('═', 64));
    // Centered section title.
    println!("{:^64}", "PREDICTION");
    // Another border.
    println!("{}", line('═', 64));

    // Show the original message text.
    println!("Message:");
    println!("  {}", message);
    println!();

    // Show the exact feature vector the model used.
    println!("Feature Vector:");
    println!(
        "  [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        input[0], input[1], input[2], input[3], input[4], input[5]
    );
    println!();

    // Show the numeric score.
    println!("Scam Score:");
    println!("  {:.3}", score);
    // Show a visual bar for the score.
    println!("  {}", score_bar(score, 30));
    println!();

    // Show the label.
    println!("Result:");
    println!("  {}", prediction_label(score));

    // Closing border.
    println!("{}", line('═', 64));
}

// Program entry point.
fn main() {
    // Print the intro UI.
    print_intro();

    // Create the training dataset.
    // Each item is:
    //   (feature_vector, target_score)
    // target:
    //   1.0 = likely scam
    //   0.0 = likely safe
    let training_data = vec![
        // Scam-like example: link + sensitive request + urgency + package mention + untrusted sender.
        (msg(1.0, 1.0, 1.0, 1.0, 0.0, 0.0), 1.0),

        // Scam-like example: fake job offer.
        (msg(1.0, 1.0, 1.0, 0.0, 1.0, 0.0), 1.0),

        // Generic phishing example.
        (msg(1.0, 1.0, 1.0, 0.0, 0.0, 0.0), 1.0),

        // Suspicious link and urgency, slightly less than full scam label.
        (msg(1.0, 0.0, 1.0, 0.0, 0.0, 0.0), 0.8),

        // Asking for information urgently.
        (msg(0.0, 1.0, 1.0, 0.0, 0.0, 0.0), 0.9),

        // Job message with a link, scam-like.
        (msg(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.8),

        // Trusted sender, no scam signals.
        (msg(0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.0),

        // Package mention from a trusted sender, mostly safe.
        (msg(0.0, 0.0, 0.0, 1.0, 0.0, 1.0), 0.1),

        // Job mention from a trusted sender, somewhat safe.
        (msg(0.0, 0.0, 0.0, 0.0, 1.0, 1.0), 0.2),

        // Neutral unknown text.
        (msg(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.1),
    ];

    // Create the neural network.
    let mut net = NeuralNet::new();

    // Train it on the dataset for 10,000 epochs using learning rate 0.5.
    net.train(&training_data, 10_000, 0.5);

    // Create a list of test messages to run after training.
    let tests = [
        (
            "Hey, are we still meeting for dinner tonight?",
            msg(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "USPS: Your package is on hold. Pay redelivery fee here.",
            msg(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        ),
        (
            "Remote job offer! Earn daily money now. Click here.",
            msg(1.0, 1.0, 1.0, 0.0, 1.0, 0.0),
        ),
        (
            "Your bank fraud team: reply now to verify your account.",
            msg(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        ),
    ];

    // Print a heading before showing test results.
    println!();
    println!("{}", line('═', 64));
    println!("{:^64}", "RUNNING TEST MESSAGES");
    println!("{}", line('═', 64));

    // Loop through each test case.
    for (label, input) in tests {
        // Get the neural network's prediction score.
        let score = net.predict(input);

        // Print a formatted result block for that message.
        print_prediction(label, input, score);
    }

    // Print a final closing section.
    println!();
    println!("{}", line('═', 64));
    println!("{:^64}", "DONE");
    println!("{}", line('═', 64));
}