# Object-Oriented Options Pricing Framework

## Overview
This repository hosts an object-oriented Python framework designed to price vanilla options, including European and American calls and puts, with a particular focus on incorporating discontinuous, point-based dividends—a key challenge in options pricing. The framework leverages object-oriented programming (OOP) principles to structure the pricing models, making it both versatile and easy to extend.

## Features
**Comprehensive Pricing Models**: Supports pricing for European and American options, with an emphasis on accurately handling discontinuous dividends.

**Modular Design**: Utilizes OOP practices with classes and functions organized across four main files, alongside a main file for easy execution and testing.

**Computation of Greeks**: Computes most important Greeks like Delta, Gamma, Theta, Vega... Which are very useful to understand how to hedge your position.

**Adjustable Parameters**: Allows for the customization of various parameters including dividends, strike prices, volatility, and more, to tailor the pricing to specific needs.

**Performance Analysis**: Includes functions for analyzing the behavior of the pricing model, such as:
*Error measurement against known pricing benchmarks.*
*Code complexity analysis based on the number of steps for precision.*
*Execution time calculation as a function of step count.*
*Comparison between the model's pricing, Black-Scholes model pricing, and variations by strike price.*

**Visualization**: Features the ability to visualize the price tree of the underlying asset, offering intuitive insights into the pricing process.
 
## Getting Started
To get started with this framework, clone the repository and ensure you have Python installed on your machine. The main.py file serves as the entry point, demonstrating how to instantiate and utilize the various pricing models and features provided.

You can modify main.py to adjust parameters, select different options types, or incorporate additional analysis as needed.

## Technical Details
The framework is divided into several key components:

Main File: Orchestrates the execution and showcases example usage.
Model Files: Contain the classes and functions for options pricing, parameter adjustment, and performance analysis.
Each option type and analysis feature is encapsulated within its own class, promoting encapsulation and ease of extension.
