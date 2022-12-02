var assert = require('assert');
var draw = require('../lib/draw');

describe('draw', function() {
    it('should correctly write file', function() {
        return draw('./test.jpeg', {
            title: "The Swift Programming Language 中文版",
            author: "Samy Pessé"
        });
    });
});

