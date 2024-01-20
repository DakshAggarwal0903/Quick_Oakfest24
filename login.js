const username = document.getElementById('username')
const password = document.getElementById('password')
const form = document.getElementById('form')
form.addEventListener('submit', function(e){
    e.preventDefault();
    const userNameValue = username.value;
    const passwordValue = password.value;               
    localStorage.setItem('username', userNameValue);
    localStorage.setItem('password',passwordValue)   
    window.location.href = 'mainpage.html';

})

