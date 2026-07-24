# NOAA GOES on AWS

Data from NOAA's GOES-R series satellite is available on Amazon S3. Tutorial on : https://github.com/NOAA-Big-Data-Program/nodd-data-docs/blob/main/GOES/

## Accessing GOES Data on AWS

You can find much [more detailed information about GOES-R Series data from NOAA](http://www.goes-r.gov/).

Examples of how to access the objects via the AWS CLI can be seen below.

`aws s3 ls noaa-goes16 --no-sign-request`

`aws s3 ls noaa-goes17 --no-sign-request`

`aws s3 cp s3://noaa-goes16/<Product>/<Year>/<Day of Year>/<Hour>/<Filename> . --no-sign-request`


## About the Data
All data files from GOES-16 (formerly GOES-R) & GOES-17 are provided in netCDF4 format. The GOES-16 data is hosted in the `noaa-goes16` Amazon S3 bucket in the us-east-1 AWS region. The GOES-17 data is hosted in the `noaa-goes17` Amazon S3 bucket in the us-east-1 AWS region. Individual files are availabe in the netCDF format with the following schema:

`<Product>/<Year>/<Day of Year>/<Hour>/<Filename>`

where:

- `<Product>` is the product generated from one of the sensors aboard the satellite (e.g.)
- `<Year>` is the year the netCDF4 file was created
- `<Day of Year>` is the numerical day of the year (1-365)
- `<Hour>` is the hour the data observation was made
- `<Filename>` is the name of the file containing the data. These are compressed and encapsulated using the netCDF4 standard.

A `<Filename>` is delineated by underscores '_' and looks like this:

`OR_ABI-L1b-RadF-M3C02_G16_s20171671145342_e20171671156109_c20171671156144.nc`

where:

- `OR`: Operational system real-time data
- `ABI-L1b-RadF-M3C02` is delineated by hyphen '-':
  - `ABI`: is ABI Sensor
  - `L1b`: is processing level, L1b data or L2
  - `Rad`: is radiances. Other products include CMIP (Cloud and Moisture Imagery products) and MCMIP (multichannel CMIP).
  - `F`: is full disk (normally every 15 minutes), C is continental U.S. (normally every 5 minutes), M1 and M2 is Mesoscale region 1 and region 2 (usually every minute each)
  - `M3`: is mode 3 (scan operation), M4 is mode 4 (only full disk scans every five minutes – no mesoscale or CONUS)
  - `C02`: is channel or band 02, There will be sixteen bands, 01-16
- `G16`: is satellite id for GOES-16 (future G17)
- `s20171671145342`: is start of scan time
  - 4 digit year
  - 3 digit day of year
  - 2 digit hour
  - 2 digit minute
  - 2 digit second
  - 1 digit tenth of second
- `e20171671156109`: is end of scan time
- `c20171671156144`: is netCDF4 file creation time
- `.nc` is netCDF file extension
