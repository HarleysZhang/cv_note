Cover Generation for GitBook
================

Generate a cover for the book.

## How to use it:

To use this plugin in your book, add it to your `book.json`:

```js
{
    "plugins": ["autocover"],
    "pluginsConfig": {
        "autocover": {
            // Configuration for autocover (see below)
        }
    }
}
```

And run `gitbook install` to fetch and prepare all plugins.

## Installation of `canvas`

This module use [node-canvas](https://github.com/LearnBoost/node-canvas). You need to install some modules on your system before being able to use it: [Wiki of node-canvas](https://github.com/LearnBoost/node-canvas/wiki/_pages).


## Configuration

Here is default configuration of **autocover**, you can change it in your book.json:

```js
{
    "title": "My Book",
    "author": "Author",
    "pluginsConfig": {
        "autocover": {
            "font": {
                "size": null,
                "family": "Impact",
                "color": "#FFF"
            },
            "size": {
                "w": 1800,
                "h": 2360
            },
            "background": {
                "color": "#09F"
            }
        }
    }
}
```
