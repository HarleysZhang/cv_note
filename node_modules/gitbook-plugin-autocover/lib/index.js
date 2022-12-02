var fs = require('fs');
var path = require('path');
var _ = require('lodash');
var Q = require('q');

var createCover = require('./draw');
var resize = require('./resize');

// Copy a file
function copy(from, to) {
    var d = Q.defer();
    var r = fs.createReadStream(from);
    var w = fs.createWriteStream(to);

    w.on('finish', d.resolve);
    w.on('error', d.reject);
    r.on('error', d.reject);

    r.pipe(w);

    return d.promise;
}

module.exports = {
    book: {},
    hooks: {
        "finish:before": function() {
            var that = this;
            var multiLangs = that.isMultilingual();
            var pluginConfig = that.config.get('pluginsConfig.autocover', {});

            var inputDir = that.resolve('.');
            var outputDir = that.output.resolve('.');

            if (multiLangs) {
                inputDir = path.resolve(inputDir, "..");
                outputDir = path.resolve(outputDir, "..");
            }

            return Q()

            // Generate big cover
            .then(function() {
                // Check if a cover already exists in the output
                if (fs.existsSync(path.join(outputDir, "cover.jpg"))) return Q();

                // Check if a cover already exists in the input
                if (fs.existsSync(path.join(inputDir, "cover.jpg"))) {
                    // Copy this cover
                    return copy(
                        path.join(inputDir, "cover.jpg"),
                        path.join(outputDir, "cover.jpg")
                    );
                }

                return createCover(
                    path.join(outputDir, "cover.jpg"),
                    _.extend({}, {
                        title: that.config.get('title'),
                        author: that.config.get('author')
                    }, pluginConfig)
                );
            })

            // Generate small cover
            .then(function() {
                // Check if a cover already exists in the output
                if (fs.existsSync(path.join(outputDir, "cover_small.jpg"))) return Q();

                // Check if a cover already exists in the input
                if (fs.existsSync(path.join(inputDir, "cover_small.jpg"))) {
                    // Copy this cover
                    return copy(
                        path.join(inputDir, "cover_small.jpg"),
                        path.join(outputDir, "cover_small.jpg")
                    );
                }

                return resize(
                    path.resolve(outputDir, "cover.jpg"),
                    path.join(outputDir, "cover_small.jpg"),
                    {
                        width: 200
                    }
                );
            })

            // Ignore error
            .fail(function(err) {
                console.log("Error with autocover: ", err.stack || err.message || err);
                return Q();
            });
        }
    }
};
