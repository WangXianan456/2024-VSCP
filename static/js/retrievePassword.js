function verifyUser() {
    var username = document.getElementById('username').value;
    var account = document.getElementById('account').value;

    $.ajax({
        url: '/forgot-password',
        method: 'POST',
        data: {
            username: username,
            account: account
        },
        success: function(response) {
            console.log("成功响应:", response); // 打印响应内容看看是否符合预期
            var data = JSON.parse(response);
            if (data.success) {
                openResetPasswordDialog(data.user_id);
            } else {
                document.getElementById('message').textContent = data.error;
            }
        },
        error: function(xhr, status, error) {
            console.log("AJAX 请求失败:", xhr, status, error); // 输出错误详情
            document.getElementById('message').textContent = "请求失败，请稍后重试";
        }
    });
}

function openResetPasswordDialog(user_id) {
    // 实现一个弹窗让用户输入新密码，并发送到 ResetPassword 接口
    var newPassword = prompt("请输入您的新密码:");
    if (newPassword) {
        $.ajax({
            url: '/reset-password',
            method: 'POST',
            data: {
                user_id: user_id,
                new_password: newPassword
            },
            success: function(response) {
                var data = JSON.parse(response);
                alert(data.message);
            },
            error: function() {
                alert("密码重置失败，请稍后重试");
            }
        });
    }
}