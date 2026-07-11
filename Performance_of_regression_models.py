# ========================================================================================
# اضافه کردن کتابخانه ها

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import  PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# ========================================================================================
# پیش پردازش داده ها 
# ========================================================================================

# خواندن فایل 
df = pd.read_csv("used_cars.csv")

# برسی کردن فایل و اصلاعات  

print(df.describe())

print('*'*119)

print(df.head())

print('*'*119)

print(df.columns)


print('*'*119)

print('Check whether the DataFrame is empty or not : ',df.empty)

# تشخیص و پردازش مقادیر گمشده

print('*'*119)

print('Detection and processing of missing values :\n',df.isnull().sum())

print('*'*119)

# تبدیل متن به عدد
df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

# جایگزینی پردازش مقادیر گمشده 
df['mileage'] = df['mileage'].fillna(df['mileage'].median())
df['engine'] = df['engine'].fillna(df['engine'].median())
df['max_power'] = df['max_power'].fillna(df['max_power'].median())
df['seats'] = df['seats'].fillna(df['seats'].median())

df.drop('torque', axis=1, inplace=True)


print("Missing values after processing:\n")
print(df.isnull().sum())

# تشخیص داده های پرت 

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers(df, 'selling_price')
df = remove_outliers(df, 'km_driven')
df = remove_outliers(df, 'mileage')
df = remove_outliers(df, 'engine')
df = remove_outliers(df, 'max_power')
df = remove_outliers(df, 'seats')

print('*'*119)


# ========================================================================================
# مصور سازی داده ها 
# ========================================================================================

# هیستوگرام
plt.figure(figsize=(8,5))
plt.hist(df['selling_price'], bins=30)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

# نمودار جعبه ای 
plt.figure(figsize=(8,5))
plt.boxplot(df['selling_price'])
plt.title('Boxplot of Selling Price')
plt.show()

# نمودار پراکندگی 
plt.figure(figsize=(8,5))
plt.scatter(df['km_driven'], df['selling_price'])
plt.title('Selling Price vs Km Driven')
plt.xlabel('Km Driven')
plt.ylabel('Selling Price')
plt.show()

# نمودار همبستگی
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ========================================================================================
# تقسیم داده ها به اموزش و تست
# ========================================================================================
# تبدیل متن به عدد
df = pd.get_dummies(
     df,
    columns=['fuel', 'seller_type', 'transmission', 'owner'],
    drop_first=True
    )



# حذف ستون name
df.drop('name', axis=1, inplace=True)


 # تعیین ویژگی‌ها و متغیر هدف
x = df.drop('selling_price', axis=1)
y = df['selling_price']

# تقسیم به داده اموزشی و تست
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("Training set:", x_train.shape)
print("Testing set :", x_test.shape)


 # ========================================================================================
# اعمال مدل های رگرسیون بر روی داده های اموزش و ارزیابی عملکرد مدل ها  
# ========================================================================================
print('*'*119)
# رگرسیون تک خطی
x_train_simple = x_train[['km_driven']]
x_test_simple = x_test[['km_driven']]

model_simple = LinearRegression()

model_simple.fit(x_train_simple, y_train)

y_pred_simple = model_simple.predict(x_test_simple)
print('Single linear regression : \n' , y_pred_simple)

# ارزیابی عملکرد مدل مدل 
print('*'*119)
print("Single Linear Regression")

mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("mse_simple :", mse_simple)
print("r2_simple :", r2_simple)

# -------------------------------------------------

# رگرسیون چند متغیره

model_multiple = LinearRegression()

model_multiple.fit(x_train, y_train)

y_pred_multiple = model_multiple.predict(x_test)
print('*'*119)
print('Multiple Linear Regression Forecasting : \n' ,y_pred_multiple)
print('*'*119)

# ارزیابی عملکرد مدل مدل 

print("Multiple Linear Regression")

mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print("mse_multiple :", mse_multiple)
print("r2_multiple :", r2_multiple)

# -------------------------------------------------


# رگرسیون چند جمله ای 

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(x_train)
X_test_poly = poly.transform(x_test)

model_poly = LinearRegression()

model_poly.fit(X_train_poly, y_train)

y_pred_poly = model_poly.predict(X_test_poly)
print('*'*119)
print('Polynomial Regression Forecasting',y_pred_poly)


print('*'*119)
print("Polynomial Regression")
# ارزیابی عملکرد مدل مدل 
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("mse_poly :", mse_poly)
print("r2_poly  :", r2_poly)

# -------------------------------------------------


# Ridge Regression

ridge = Ridge(alpha=1)

ridge.fit(x_train, y_train)

y_pred_ridge = ridge.predict(x_test)

print('*'*119)
print(' Ridge prediction: ',y_pred_ridge )

print('*'*119)
print("Ridge Regression")
# ارزیابی عملکرد مدل مدل
mse_ridge  = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("mse_ridge  :",mse_ridge )
print("r2_ridge  :", r2_ridge)

# -------------------------------------------------

# Lasso Regression

lasso = Lasso(alpha=1)

lasso.fit(x_train, y_train)

y_pred_lasso = lasso.predict(x_test)
print('*'*119)
print(' Lasso prediction: ',y_pred_lasso )

print('*'*119)
print("Lasso Regression")
# ارزیابی عملکرد مدل مدل
mse_lasso  = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("mse_lasso  :", mse_lasso )
print("r2_lasso :", r2_lasso)

# -------------------------------------------------

# ElasticNet Regression

elastic = ElasticNet(alpha=1, l1_ratio=0.5)

elastic.fit(x_train, y_train)

y_pred_elastic = elastic.predict(x_test)

print('*'*119)
print(' ElasticNet prediction: ',y_pred_elastic )

print('*'*119)
print("ElasticNet Regression")
# ارزیابی عملکرد مدل مدل
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)

print("mse_elastic :", mse_elastic)
print("r2_elastic:", r2_elastic)

# ========================================================================================
# مقایسه مدل ها با نمودار  
# ========================================================================================

models = ['Simple', 'Multiple', 'Polynomial', 'Ridge', 'Lasso', 'ElasticNet']

mse_values = [
    mse_simple,
    mse_multiple,
    mse_poly,
    mse_ridge,
    mse_lasso,
    mse_elastic
]

r2_values = [
    r2_simple,
    r2_multiple,
    r2_poly,
    r2_ridge,
    r2_lasso,
    r2_elastic
]
# -------------------------------------------------
# نمودار مقایسه MSE

plt.figure(figsize=(8,5))

plt.bar(models, mse_values)

plt.title('Comparison of MSE for Regression Models')
plt.xlabel('Regression Models')
plt.ylabel('MSE')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()
# -------------------------------------------------
# نمودار مقایسه R2

plt.figure(figsize=(8,5))

plt.bar(models, r2_values)

plt.title('Comparison of R2 Score for Regression Models')
plt.xlabel('Regression Models')
plt.ylabel('R2 Score')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()

