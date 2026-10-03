const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");

form.addEventListener("submit", async (event) => {
  // Prevent the browser from reloading the page.
  event.preventDefault();

  const message = input.value.trim();

  if (!message) {
    return;
  }

  // Show user's message immediately.
  addMessage(message, "user");

  // Clear input.
  input.value = "";

  try {
    // Send the message to Flask.
    const response = await fetch("/api/chat", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        message: message,
      }),
    });

    if (!response.ok) {
      throw new Error("Server returned an error.");
    }

    // Convert Flask JSON response into a JS object.
    const data = await response.json();

    // Display response from Python.
    addMessage(data.response, "bot");
  } catch (error) {
    console.error(error);

    addMessage("Something went wrong. Please try again.", "bot");
  }
});

function addMessage(text, sender) {
  const element = document.createElement("div");

  element.classList.add("message", sender);

  // textContent is preferable to innerHTML here because
  // user/backend text should not be interpreted as HTML.
  element.textContent = text;

  messages.appendChild(element);

  // Scroll to newest message.
  messages.scrollTop = messages.scrollHeight;
}
