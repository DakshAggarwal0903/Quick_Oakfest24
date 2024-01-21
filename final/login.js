const username = document.getElementById('username')
const password = document.getElementById('password')
const email = document.getElementById('email')
const form = document.getElementById('form')
form.addEventListener('submit', function(e){
    e.preventDefault();
    const userNameValue = username.value;
    const passwordValue = password.value;
    const emailValue = email.value;               
    localStorage.setItem('username', userNameValue);
    localStorage.setItem('password',passwordValue);
    localStorage.setItem('email',emailValue)   
    window.location.href('homepage.html')

})

