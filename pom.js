foam.POM({
  name: 'aILab',
  excludes: [ '*' ],
  projects: [
    { name: 'foam3/pom'},
    { name: 'src/com/foam/ailab/pom'},
    { name: 'journals/pom' }
  ],
  licenses: `
    // Add your license header here
  `,
  envs: {
    version: '1.0.0',
    // javaMainArgs: 'spid:aILab'
  },
  tasks: [
    function javaManifest() {
      JAVA_MANIFEST_VENDOR_ID = 'ailab.foam.com';
    }
  ]
});
