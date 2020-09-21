# PyTorch Parquet Data Loader
This library holds a number of classes which help reading data from Parquet files into the PyTorch ecosystem easily!
Although this library is intended to be used for natural language processing projects and with NLP libraries, it is extremely flexible.
Feel free to use, modify or fork this library in any way!

## Supported Libraries
| Library                                                                    | Requirements                                                                                                              | Usage Notes                                                                                                                                                            |
|----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) | Requires [PyArrow](https://arrow.apache.org/docs/python/) and (optionally) [Petastorm](https://github.com/uber/petastorm) | The more basic [PyArrow](https://arrow.apache.org/docs/python/) implementation is far easier to understand, but not battle tested.                                     |
| [Transformers](https://huggingface.co/transformers/)                       |                                                                                                                           | Can be used with either PyTorch-Lightning implementation, but Petastorm casts data types from one format to another several times midway, which can impare performance |
| [AllenNLP](https://allennlp.org)                                           | [PyArrow](https://arrow.apache.org/docs/python/)                                                                          | Not implemented yet                                                                                                                                                    |

Please look [here](https://github.com/uber/petastorm/issues/603) for further information on using Petastorm with Hugging Face Transformers.

## Difference from Petastorm
Petastorm is a great (albeit complex) library for using Parquet files in a large variety of situations.
Although they have basic PyTorch support, their solution is tough to understand.
This can make it difficult to debug and modify for personal use.

Alternatively, `PyParquetLoaders` is focused on providing Python classes which are easy to use, understand and modify.
This means anyone can get started with their PyTorch models reasonably quickly, even if they're doing something slightly different/unique.
Currently PyParquetLoaders supports PyTorch-Lightning and Hugging Face Transformers, with AllenNLP support comming soon!


## Usage Guide
To use a Parquet file for training a PyTorch model simply choose and import the right data set/loader (for your library of choice).
Then you can simply try and use it just as you would for any other (simple) text/image file (look at your libraries relavent docs).
Some examples are included in the [Tests.py](Tests.py) script.
