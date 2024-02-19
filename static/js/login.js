const loginForm = document.querySelector('#login-form')
const username = document.querySelector('#login-form input[type="text"]')
const password = document.querySelector('#login-form input[type="password"]')
const submitButton = document.querySelector('#login-form#login-form button[type="submit"]')

loginForm.addEventListener('input', checkFields)


function checkFields() {
    if (username.value === "" || password.value === "") {
        submitButton.disabled = true
        return true
    } else {
        submitButton.disabled = false
        return false
    }
}



