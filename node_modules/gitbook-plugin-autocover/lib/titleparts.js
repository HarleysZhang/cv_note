var fontSize = require('./fontsize');

// titleParts breaks a string into a list of lines
// where each line fits inside the box of dimensions = width*height
// for a given font family
function titleParts(title, font, width, height) {
    // Words of title
    var parts = title.split(/\s+/);

    return parts
    .reduce(function(lines, part) {
        // First part
        if(lines.length === 0) return [part];

        // Last processed part
        var prevPart = lines[lines.length - 1];
        // Current part appended to last part
        var newPart = prevPart + ' ' + part;

        // Size of previous part by itself
        var fsize = fontSize(
            prevPart, font,
            width, height
        );

        // How big is it if we add our new part ?
        var fsize2 = fontSize(
            newPart, font,
            width, height
        );

        // If sizes are the same, then merge parts to same line
        if(fsize == fsize2 && fsize2) {
            lines[lines.length - 1] = newPart;
            return lines;
        }

        return lines.concat(part);
    }, []);
}

module.exports = titleParts;
