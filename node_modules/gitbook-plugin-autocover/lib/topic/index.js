var topics = require('./topics.json');

// Break up the incomming text into tokens
function tokenize(str) {
    return str.toLowerCase().split(/\s+/).filter(Boolean);
}

// Intersection of two arrays
function intersect(a, b) {
    var t;
    // Swap a and b (to use shortest for loop iteration)
    if (b.length > a.length) {
        t = b, b = a, a = t;
    }

    return a.filter(function (e) {
        if (b.indexOf(e) !== -1) return true;
    });
}

function doIntersect(a, b) {
    return intersect(a, b).length > 0;
}

// Derive topic based on keywords in a string
// this is meant for small corpuses of text (titles, etc ...)
module.exports = function topic(str) {
    var tokens = tokenize(str);

    return Object.keys(topics).filter(function(key) {
        var topicTokens = topics[key];

        return (
            // Exact match of topic name
            tokens.indexOf(key) !== -1 ||

            // Match of topic tokens
            doIntersect(tokens, topicTokens)
        );
    });
};
