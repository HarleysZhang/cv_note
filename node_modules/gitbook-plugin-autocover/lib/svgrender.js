var Q = require('q');
var fs = require('fs');

var canvg = require('canvg');
var Canvas = require('canvas');

function renderSvg(outputfilename, svgdata, width, height) {
    var d = Q.defer();

    try {
        var canvas = new Canvas(width, height);

        // Render SVG image
        canvg(canvas, svgdata,
            {
                ignoreAnimation: true,
                renderCallback: function()
                {
                    // Create streams
                    var out = fs.createWriteStream(outputfilename);
                    var stream = canvas.jpegStream();

                    // Pipe
                    stream.pipe(out);

                    // Wait till finished piping/writing
                    out.on('close', d.resolve);
                    out.on('error', d.reject);
                }
        });
    } catch(err) {
        return Q.reject(err);
    }

    return d.promise;
}

module.exports = renderSvg;
