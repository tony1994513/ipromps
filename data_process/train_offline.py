#!/usr/bin/python
import load_data
import train_models
import noise_cov_cal

def main():
    print("-----------")
    print('## Running the %s' % load_data.__name__)
    load_data.main()
    print("-----------")
    print('## Running the %s' % noise_cov_cal.__name__)
    noise_cov_cal.main()
    print("-----------")
    print('## Running the %s' % train_models.__name__)
    train_models.main()

if __name__ == '__main__':
    main()
