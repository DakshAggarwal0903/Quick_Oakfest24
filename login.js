const username = document.getElementById('username')
const form = document.getElementById('form')
form.addEventListener('submit', function(e){
    //add event listener to listen for certain commands
    e.preventDefault();//prevents normal processing of form
    const userNameValue = username.value;                
    localStorage.setItem('username', userNameValue);   
    window.location.href = 'dashboard.html';
})