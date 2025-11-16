import matplotlib.pyplot as plt
data = {1: 18256432, 2: 10979213, 3: 17809027, 4: 37360696, 5: 148649950}
negative = data[1] + data[2]
neutral = data[3]
positive = data[4] + data[5]
sentiment_data = {"Negative": negative, "Neutral": neutral, "Positive": positive}

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 3))
ax[0].bar(data.keys(), data.values())
ax[0].set_xlabel('Number of stars')
ax[0].set_ylabel('Number of reviews (in millions)')
ax[0].set_yticks([0, 25000000, 50000000, 75000000, 100000000, 125000000, 150000000, 175000000, 200000000],
                 ['0', '25', '50', '75', '100', '125', '150', '175', '200'])
ax[0].set_title('Amazon Product Reviews Ratings')
ax[1].bar(sentiment_data.keys(), sentiment_data.values())
ax[1].set_xlabel('Sentiment')
ax[1].set_title('Amazon Product Reviews Sentiment')
plt.tight_layout()
plt.savefig('./results/all_review_distribution.png')

