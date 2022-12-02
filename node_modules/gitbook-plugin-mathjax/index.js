<<<<<<< HEAD
var Promise = require('q');
var crc     = require('crc');
var mjAPI   = require('mathjax-node/lib/mj-single.js');

var started   = false;
=======
var Q = require('q');
var fs = require('fs');
var path = require('path');
var crc = require('crc');
var exec = require('child_process').exec;
var mjAPI = require('mathjax-node/lib/mj-single.js');

var started = false;
>>>>>>> master
var countMath = 0;
var cache     = {};

/**
    Prepare MathJaX
*/
function prepareMathJax() {
    if (started) {
        return;
    }

    mjAPI.config({
        MathJax: {
            SVG: {
                font: 'TeX'
            }
        }
    });
    mjAPI.start();

    started = true;
}

/**
    Convert a tex formula into a SVG text

    @param {String} tex
    @param {Object} options
    @return {Promise<String>}
*/
function convertTexToSvg(tex, options) {
    var d = Promise.defer();
    options = options || {};

    prepareMathJax();

    mjAPI.typeset({
        math:         tex,
        format:       (options.inline ? 'inline-TeX' : 'TeX'),
        svg:          true,
        speakText:    true,
        speakRuleset: 'mathspeak',
        speakStyle:   'default',
        ex:           6,
        width:        100,
        linebreaks:   true
    }, function(data) {
        if (data.errors) {
            return d.reject(new Error(data.errors));
        }

        d.resolve(options.write ? null : data.svg);
    });

    return d.promise;
}

/**
    Process a math block

    @param {Block} block
    @return {Promise<props>}
*/
function processBlock(block) {
    var book     = this;
    var tex      = block.children;
    var isInline = !(tex[0] == '\n');
    var config   = book.config.get('pluginsConfig.mathjax', {});

    return Promise()
    .then(function() {
        // For website return as script
        if ((book.output.name == 'website' || book.output.name == 'json')
            && !config.forceSVG) {
            return {
                isSVG:   false,
                content: block.children,
                inline:  isInline
            };
        }

        // Get key for cache
        var hashTex = crc.crc32(tex).toString(16);

        // Compute SVG filename
        var imgFilename = '_mathjax_' + hashTex + '.svg';

        return Promise()
        .then(function() {
            // Check if not already cached
            if (cache[hashTex]) {
                return;
            }

            cache[hashTex] = true;
            countMath = countMath + 1;

            return convertTexToSvg(tex, { inline: isInline })
            .then(function(svg) {
                return book.output.writeFile(imgFilename, svg);
            });
        })
        .then(function() {
            return {
                isSVG:    true,
                filename: imgFilename,
                inline:   isInline
            };
        });
    });
}

module.exports = {
    blocks: {
        math: {
            shortcuts: {
                parsers: [ 'markdown', 'asciidoc' ],
                start:   '$$',
                end:     '$$'
            },
            process: processBlock
        }
    }
};
