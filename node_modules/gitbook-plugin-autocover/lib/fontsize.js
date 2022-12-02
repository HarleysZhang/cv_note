var Canvas = require('canvas');

function textsize(str, size, font) {
    // We only want the context to do our predictions
    var ctx = new Canvas().getContext('2d');

    // Set font
    ctx.font = size+"px "+font;

    // Get dimensions it would occupy
    var dim = ctx.measureText(str);

    return {
        width: dim.width,
        height: dim.emHeightAscent + dim.emHeightDescent,
    };
}

// Get the good font size for text to fit in a given width
function fontSizeForDimensions(str, font, width, height, lower, upper) {
    // Lower and upper bounds for font
    lower = (lower === undefined) ? 0      : lower;
    upper = (upper === undefined) ? height : upper;

    // The font size we're guessing with
    var middle = Math.floor((upper + lower) / 2);

    if(middle === lower) {
        return middle;
    }

    // Get text dimensions
    var tsize = textsize(str, middle, font);

    return (
        // Are we above or below ?
        (tsize.width <= width && tsize.height <= height) ?
        // Go up
        fontSizeForDimensions(str, font, width, height, middle, upper) :
        // Go down
        fontSizeForDimensions(str, font, width, height, lower, middle)
    );
}

module.exports = fontSizeForDimensions;
