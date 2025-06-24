function checkLogin() {
    $.ajax({
        url: '/CheckLogin',
        method: 'GET',
        success: function(response) {
            try {
                response = JSON.parse(response);
                if (response.logged_in) {
                       window.open('FisherHistory.html', '_blank');
                } else {
                    document.getElementById('errorModal').style.display = 'block';
                }
            } catch (e) {
                console.error('Error parsing response:', e);
            }
        },
        error: function() {
            alert('未登录账号！无法显示历史记录！');
        }
    });
}