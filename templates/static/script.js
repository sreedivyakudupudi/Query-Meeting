function generateChatbotResponse(userMessage) {
  const data = { "user_input": userMessage };  // Adjust the key based on the expected format
  return fetch("/interact", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => {
      if (response.ok) {
        return response.text();
      }
      throw new Error(`Error fetching chatbot response. Status: ${response.status}`);
    })
    .catch((error) => {
      console.error("Error fetching chatbot response:", error.message);
    });
}


function submitQuestion() {
  const question = document.getElementById("question").value;
  console.log(question);
  if (question.length > 0) {
    let userMessage = `<div class="chat message">
            <img src="/static/user.png">
            <span>${question}</span>
          </div>`;
    document
      .querySelector(".message-box")
      .insertAdjacentHTML("beforeend", userMessage);

    generateChatbotResponse(question).then((decoded_output) => {
      let jsonResponse = JSON.parse(decoded_output);
      let answer = jsonResponse.answer;

      let chatbotResponse = `<div class="chat response">
            <img src="/static/chatbot.png">
            <span>${answer}</span>
          </div>`;
      document
        .querySelector(".message-box")
        .insertAdjacentHTML("beforeend", chatbotResponse);
    });
  }
}

fetch("/landing", {
  method: "GET",
  headers: {
    "Content-Type": "application/json",
  },
})
  .then((response) => {
    if (response.ok) {
      console.log("Upload Video Page");
    } else {
      throw new Error(`Error fetching /landing. Status: ${response.status}`);
    }
  })
  .catch((error) => {
    console.error("Error fetching /landing:", error.message);
  });
