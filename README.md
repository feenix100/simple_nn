# 🧠 Rust Neural Network – Scam Text Detection

A simple, from-scratch neural network written in Rust that classifies text messages as **likely scam** or **likely safe**.

This project is designed to teach neural network fundamentals while solving a real-world problem using a lightweight, explainable model.

---

## 🚀 What This App Does

Instead of processing raw text directly, this model converts messages into a small set of numeric features, such as:

- Contains a link  
- Requests sensitive information  
- Uses urgent language  
- Mentions package delivery  
- Mentions job opportunity  
- Comes from a trusted sender  

These features are fed into a neural network:

6 inputs → 4 hidden neurons → 1 output (scam score)

The output is a value between 0 and 1:
- 0.0 → likely safe  
- 1.0 → likely scam  

---

## 📸 Screenshot

![App Screenshot](screenshot.png)

(Add your screenshot file to the repo root or update the path)

---

## 🧠 Concepts Covered

- Neural network fundamentals
- Feature engineering
- Forward propagation
- Backpropagation (gradient descent)
- Sigmoid activation
- Struct-based design in Rust
- Working with arrays and loops in Rust

---

## 📂 Project Structure

simple_nn/
├── Cargo.toml
└── src/
    └── main.rs

---

## ⚙️ How to Run

### 1. Clone the repository

git clone https://github.com/yourusername/your-repo.git
cd your-repo

### 2. Build and run

cargo run

---

## 🧪 Example Output

Message: USPS: Your package is on hold. Pay redelivery fee here.  
Scam score: 0.923  
Prediction: likely scam  

Message: Hey, are we still meeting for dinner tonight?  
Scam score: 0.102  
Prediction: likely safe  

---

## 🏗️ How It Works (High-Level)

### 1. Input Encoding
Each message is converted into a vector:

[has_link, asks_sensitive, urgent, package, job, trusted_sender]

### 2. Forward Pass
- Inputs are multiplied by weights
- Bias is added
- Activation function (sigmoid) is applied
- Output becomes a probability-like score

### 3. Training
- The model predicts a result
- Error is calculated
- Weights are adjusted
- Repeated thousands of times

---

## ⚠️ Limitations

This is a toy model for learning purposes:

- Does NOT understand raw text
- Uses handcrafted features instead of NLP
- Small dataset
- Not production-ready

---

## 🔮 Future Improvements

- Parse real text into features automatically
- Add more training data
- Expand network architecture
- Add CLI input for live predictions
- Integrate with a Rust ML framework

---

## 📜 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome. Feel free to open an issue with ideas or improvements.
