function redirectToMainPage() {
    // Dummy check for username and password (you should implement proper authentication)
    var username_prompt = document.getElementById('username').value;
    var password_prompt = document.getElementById('password').value;
    const username = 'user'
    const password = 'pass'
    // Example: Redirect to the main page if username and password are not empty
    if (username_prompt == username && password_prompt == password) {
      window.location.href = 'mainpage.html';
    } else {
      alert('Please enter valid credentials.');
    }
  }

