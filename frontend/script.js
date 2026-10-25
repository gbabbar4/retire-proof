
const form = document.getElementById("retirement-form");

const output = document.getElementById("prediction-output");

// Address of our FastAPI backend
const API_URL = "http://127.0.0.1:8000/predict";

form.addEventListener("submit", async function(event) {

    // Stop page from refreshing
    event.preventDefault();

    // Collect 7 inputs from the HTML form
    const data = {
        age: Number(document.getElementById("age").value),
        retirement: Number(document.getElementById("retirement").value),
        corpus: Number(document.getElementById("corpus").value),
        monthly_savings: Number(document.getElementById("monthly_savings").value),
        monthly_expense_today: Number(document.getElementById("monthly_expense_today").value),
        equity_alloc: Number(document.getElementById("equity_alloc").value) / 100,
        infl: Number(document.getElementById("infl").value) / 100
    };

    // Basic financial input check
    if (data.retirement <= data.age) {
        output.textContent =
            "Retirement age must be greater than current age.";
        return;
    }


    console.log("DATA SENT TO BACKEND:", data);


    output.textContent = "Calculating prediction...";

    try {

        // Send user inputs to our FastAPI backend
        const response = await fetch(API_URL, {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        });

        // Convert JSON response into JavaScript object
        const result = await response.json();

        // Check for backend errors
        if (!response.ok) {
            throw new Error(
                JSON.stringify(result.detail ?? "Prediction failed")
            );
        }

        // Display Retirement Survival Probability in %
        output.textContent =
            `Retirement Survival Probability: ${(result.prediction * 100).toFixed(2)}% ` +
            `(Model ${result.model_version})`;

        console.log("Backend response:", result);

    } catch (error) {

        console.error("API Error:", error);

        output.textContent =
            "Unable to get prediction: " + error.message;
    }

});
