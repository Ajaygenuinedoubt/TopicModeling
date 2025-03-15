<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topic Modeling with Articles Reviews</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* Lighting animation */
        .glowing-button {
            background-color: #403a05a7;
            color: whitesmoke;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            transition: box-shadow 0.3s ease-in-out;
        }
        .glowing-button:hover {
            box-shadow: 0 0 10px rgba(54, 8, 237, 0.7), 0 0 20px rgba(59, 130, 246, 0.7), 0 0 30px rgba(59, 130, 246, 0.7);
        }

        /* Glowing effect on topics */
        .glowing-topic {
            animation: glow 1.5s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from {
                box-shadow: 0 0 5px #c6f204;
            }
            to {
                box-shadow: 0 0 20px #d39009;
            }
        }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-8">
        <h1 class="text-4xl font-bold text-center mb-8 glowing-topic">Topic Modeling with Articles Reviews By :- Ajay Kumar Jha</h1>
        
        <!-- Input Form -->
        <form method="POST" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <div class="mb-4">
                <label class="block text-gray-700 text-sm font-bold mb-2" for="text_input">Enter Text:</label>
                <textarea id="text_input" name="text_input" rows="4" class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" placeholder="Enter a review, article,engineering or any text here..."></textarea>
            </div>
            <div class="flex items-center justify-between">
                <button class="glowing-button" type="submit">Analyze Topics</button>
            </div>
        </form>

        <!-- Display Topics -->
        {% if topics %}
        <div class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <h2 class="text-2xl font-bold mb-4 glowing-topic">Identified Topics:</h2>
            <ul class="list-disc list-inside">
                {% for topic in topics %}
                    <li class="mb-2 glowing-topic">
                        <strong>Topic {{ loop.index }}:</strong> {{ topic | join(", ") }} (Word Count: {{ topic_word_count[loop.index0] }})
                    </li>
                {% endfor %}
            </ul>
        </div>

        <!-- Word Count Graph -->
        <div class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <h2 class="text-2xl font-bold mb-4 glowing-topic">Word Countof your text  per Topic</h2>
            <canvas id="wordCountChart"></canvas>
        </div>

        <script>
            // Word count data for the chart
            var topicLabels = [{% for topic in topics %}'Topic {{ loop.index }}',{% endfor %}];
            var wordCounts = [{% for count in topic_word_count %}{{ count }},{% endfor %}];

            // Chart.js code for the word count bar chart
            var ctx = document.getElementById('wordCountChart').getContext('2d');
            var wordCountChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: topicLabels,
                    datasets: [{
                        label: 'Word Count',
                        data: wordCounts,
                        backgroundColor: 'rgba(270, 10, 800, 0.9)',
                        borderColor: 'rgba(509, 30, 746, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        </script>
        {% endif %}
    </div>
</body>
</html>
