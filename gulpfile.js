const gulp = require('gulp');
const gulpLoadPlugins = require('gulp-load-plugins');

const $ = gulpLoadPlugins();

gulp.task('publish', () => {
  console.log('Publish Gitbook (_book) to Github Pages');
  return gulp.src('./_book/**/*')
    .pipe($.ghPages({
      origin: 'origin',
      branch: 'gh-pages'
    }));
});