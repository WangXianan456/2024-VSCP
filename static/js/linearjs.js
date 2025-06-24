 var isCalculated = false;
    document.getElementById('inputForm').addEventListener('submit', function(event) {
        event.preventDefault();
        var dataset1 = parseData(document.getElementById('data1').value);
        var dataset2 = parseData(document.getElementById('data2').value);

        if (dataset1.length === 0 || dataset2.length === 0 || dataset1.length !== dataset2.length) {
            alert('请输入正确格式的数据！');
            return;
        }

        var regressionResult = calculateLinearRegression(dataset1, dataset2);
        document.getElementById('L').textContent = 'y=' + regressionResult.slope + '*x+' +regressionResult.intercept;
        document.getElementById('result').style.display = 'block';
        document.getElementById('x_avg').textContent = regressionResult.x_avg.toFixed(2);
        document.getElementById('y_avg').textContent = regressionResult.y_avg.toFixed(2);
        document.getElementById('Lxx').textContent = regressionResult.Lxx.toFixed(2);
        document.getElementById('Lyy').textContent = regressionResult.Lyy.toFixed(2);
        document.getElementById('Lxy').textContent = regressionResult.Lxy.toFixed(2);
        document.getElementById('beta1').textContent = regressionResult.slope.toFixed(2);
        document.getElementById('beta0').textContent = regressionResult.intercept.toFixed(2);
        document.getElementById('SSe').textContent = regressionResult.SSe.toFixed(2);
        document.getElementById('sigma').textContent = regressionResult.sigma.toFixed(2);
        document.getElementById('r').textContent = regressionResult.r.toFixed(2);
        document.getElementById('rr').textContent = regressionResult.rr.toFixed(2);
        isCalculated = true;
        updateResults(regressionResult);
        drawChart(dataset1, dataset2, regressionResult);
  });

    document.getElementById('saveButton').addEventListener('click', function() {
if (!isCalculated) {
    alert('请先完成计算再保存结果！');
    return;
}

var dataset1 = document.getElementById('data1').value;
var dataset2 = document.getElementById('data2').value;
var regressionResult = getRegressionResultFromPage();

$.ajax({
    url: '/saveLinearHistory',
    type: 'POST',
    data: JSON.stringify({
        data1: JSON.parse(dataset1),
        data2: JSON.parse(dataset2),
        regressionResult: regressionResult
    }),
    contentType: 'application/json',
    success: function(response) {
        alert('记录保存成功！');
        isCalculated = false;  // 重置计算状态
    },
    error: function(error) {
        alert('记录保存失败，请重试！');
    }
});
});

     function getRegressionResultFromPage() {
        return {
            x_avg: document.getElementById('x_avg').textContent,
            y_avg: document.getElementById('y_avg').textContent,
            Lxx: document.getElementById('Lxx').textContent,
            Lyy: document.getElementById('Lyy').textContent,
            Lxy: document.getElementById('Lxy').textContent,
            slope: document.getElementById('beta1').textContent,
            intercept: document.getElementById('beta0').textContent,
            SSe: document.getElementById('SSe').textContent,
            sigma: document.getElementById('sigma').textContent,
            r: document.getElementById('r').textContent,
            rr: document.getElementById('rr').textContent
        };
    }

     function updateResults(regressionResult) {
        document.getElementById('x_avg').textContent = regressionResult.x_avg.toFixed(2);
        document.getElementById('y_avg').textContent = regressionResult.y_avg.toFixed(2);
        document.getElementById('Lxx').textContent = regressionResult.Lxx.toFixed(2);
        document.getElementById('Lyy').textContent = regressionResult.Lyy.toFixed(2);
        document.getElementById('Lxy').textContent = regressionResult.Lxy.toFixed(2);
        document.getElementById('beta1').textContent = regressionResult.slope.toFixed(2);
        document.getElementById('beta0').textContent = regressionResult.intercept.toFixed(2);
        document.getElementById('SSe').textContent = regressionResult.SSe.toFixed(2);
        document.getElementById('sigma').textContent = regressionResult.sigma.toFixed(2);
        document.getElementById('r').textContent = regressionResult.r.toFixed(2);
        document.getElementById('rr').textContent = regressionResult.rr.toFixed(2);
        }

    function drawChart(dataset1, dataset2, regressionResult) {
        var ctx = document.getElementById('chart').getContext('2d');
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '数据点集',
                    data: dataset1.map(function(x, i) { return { x: x, y: dataset2[i] }; }),
                    pointBackgroundColor: 'red'
                }, {
                    label: '回归直线L',
                    data: [
                        {x: Math.min(...dataset1), y: regressionResult.intercept + regressionResult.slope * Math.min(...dataset1)},
                        {x: Math.max(...dataset1), y: regressionResult.intercept + regressionResult.slope * Math.max(...dataset1)}
                    ],
                    type: 'line',
                    pointRadius: 0,
                    borderColor: 'blue'
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
    }
