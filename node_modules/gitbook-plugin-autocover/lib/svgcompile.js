var Q = require('q');
var fs = require('fs');
var _ = require('lodash');

function compileFromFile(filename, data, callback) {
    if (typeof filename !== 'string') throw new Error('filename must be a string');
    if (typeof data !== 'object') throw new Error('data must be an object');
    if (typeof callback !== 'function') throw new Error('must provide a callback');

    fs.readFile(filename, 'utf8', function(err, contents) {
        if (err) callback(err);
        var template = _.template(contents);
        callback(null, template(data));
    });
}

function compileQ(filename, data) {
    return Q.nfcall(compileFromFile, filename, data);
}

module.exports = compileQ;
