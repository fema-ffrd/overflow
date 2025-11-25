# Overview

Overflow is a high-performance Python library designed for processing Digital Elevation Models (DEMs) at scale. The library addresses the computational challenges of hydrological terrain analysis when working with massive raster datasets that can range from local watersheds to continental-scale terrain models.

## What Overflow Does

Overflow provides a complete suite of tools for deriving hydrographic features from raw elevation data. The library handles the full processing pipeline from terrain conditioning through feature extraction. This includes removing artifacts and depressions in the elevation data, computing how water flows across the landscape, calculating drainage area, and extracting vector representations of stream networks and basin boundaries.

## Who Should Use Overflow

Overflow is designed for anyone who needs to run these hydrolocial process on large DEMs efficiently. The library is particularly valuable when working with datasets that exceed the practical limits of traditional single-threaded tools or when processing time is a critical constraint.

Users should have basic familiarity with hydrological concepts such as flow direction, flow accumulation, and drainage networks. The library assumes you are comfortable working with geospatial raster data and understand fundamental GIS concepts. Programming experience with Python is helpful for using the API, though the command-line interface provides access to core functionality without requiring code.

## Design Philosophy

Overflow aims to be scalable, performant, and correct - in that order. There are cases where correctness is sacrificed in favor of the other two so long as it is not a practical issue. In most cases, the algorithms produce results that are mathematically equivalent to authoritative methods used throughout the hydrological community. Every operation is designed to scale from laptop-sized datasets to continental DEMs without requiring specialized hardware.
