
$(document).ready(function() {
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault(); // 阻止表单的默认提交行为

        var formData = new FormData(this);
        $.ajax({
            url: $(this).attr('action'),
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            dataType: 'json',  // 确保期待的响应类型为JSON
            success: function(result) { // 直接接收解析后的JSON对象
                var resultHtml = `预测年龄: ${result.predicted_age}岁, 实际年龄: ${result.real_age}岁, 绝对误差: ${result.absolute_error}`;
                $('#predictionResults').html(resultHtml);
            },
            error: function(xhr) {
                var error = JSON.parse(xhr.responseText);
                alert('Error: ' + error.error);
            }
        });
    });
});