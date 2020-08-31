import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Import data sets
data = pandas.read_csv("summer-products-with-rating-and-performance_2020-08.csv")
keywords = pandas.read_csv("unique-categories.sorted-by-count.csv")
unsorted_keywords = pandas.read_csv("unique-categories.csv")

# Delete unnecessary columns
pre_processed_data = data.drop(columns = ["title", "title_orig", "currency_buyer",
                                "rating", "rating_count", "product_color",
                                "product_variation_size_id",
                                "shipping_option_name", "urgency_text",
                                "origin_country", "merchant_title",
                                "merchant_name", "merchant_info_subtitle",
                                "merchant_id", "merchant_profile_picture",
                                "product_url", "product_picture", "product_id",
                                "theme", "crawl_month"])

# Get total counts of keywords
sum_keywords = sum(keywords["count"])

# # Calculate every keywords rating with "count/total"
keywords["rate"] = keywords["count"] / sum_keywords

# Get evry tag for evry product
tags = pre_processed_data["tags"].str.split(pat = ",", expand=True)
tags = tags.replace(numpy.nan, '', regex=True)

s = tags.shape

# Turn keywords in tags to their own rating values
for i in range(s[0]):
    for j in range(s[1]):
        if tags[j][i] != str(''):
            tags[j][i] = keywords[keywords["keyword"].str.contains(tags[j][i])].sum()["rate"]
        else:
            tags[j][i] = 0

#Calculate and swith tags column with their total rating
pre_processed_data["tags"] = numpy.sum(tags, axis = 1)
pre_processed_data["has_urgency_banner"] = pre_processed_data["has_urgency_banner"].fillna(0)

pre_processed_data["total_merc_point"] = pre_processed_data["merchant_rating_count"] * pre_processed_data["merchant_rating"]

pre_processed_data = pre_processed_data.drop(columns = ["merchant_rating_count",
                                                        "merchant_rating"])

pre_processed_data["rating_five_count"] = pre_processed_data["rating_five_count"].fillna(0)
pre_processed_data["rating_four_count"] = pre_processed_data["rating_four_count"].fillna(0)
pre_processed_data["rating_three_count"] = pre_processed_data["rating_three_count"].fillna(0)
pre_processed_data["rating_two_count"] = pre_processed_data["rating_two_count"].fillna(0)
pre_processed_data["rating_one_count"] = pre_processed_data["rating_one_count"].fillna(0)


Y = pre_processed_data["price"]
X = pre_processed_data.drop(columns=["price"])



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

scaler_x = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)




regressor = RandomForestRegressor(n_estimators=1000)
regressor.fit(x_train, y_train)

# Use the forest's predict method on the test data
predictions = regressor.predict(x_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(numpy.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - numpy.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
