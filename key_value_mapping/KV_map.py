import re


def text_clean(text):
    text = re.sub('ः', ':', text)
    text = re.sub(':', '|', text)
    text = re.sub('\s', '-', text)
    print(text)
    return text


def find_value(text):
    driving_number = re.findall("[^\d](\d{2}[-]\d{2}[-]\d{8})[^\d]", text)
    phone_number = re.findall("[^\d](\d{10})[^\d]", text)
    dates = re.findall("[^\d][^\d](\d{2}[-]\d{2}[-]\d{4})[^\d]", text)
    print(f'Driving License Number: {driving_number}')
    print(f'Phone Number: {phone_number}')
    print(f'Dates: {dates}')
