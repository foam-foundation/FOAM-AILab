/**
 * @license
 * Copyright 2026 FOAM Foundation and contributors
 * SPDX-License-Identifier: Apache-2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 */

foam.POM({
  name: 'aILab',
  excludes: [ '*' ],
  setFlags: { ml: true }, // FOAM's AI model training features: ML Lab menus, dataset and training models
  projects: [
    { name: 'foam3/pom'},
    { name: 'src/com/foamdev/ailab/pom'},
    { name: 'journals/pom' }
  ],
  licenses: `
  /**
   * @license
   * Copyright 2026 FOAM Foundation and contributors
   * SPDX-License-Identifier: Apache-2.0
   * http://www.apache.org/licenses/LICENSE-2.0
  */
  `,
  envs: {
    version: '1.0.0',
    // javaMainArgs: 'spid:aILab'
  },
  tasks: [
    function javaManifest() {
      JAVA_MANIFEST_VENDOR_ID = 'ailab.foamdev.com';
    }
  ]
});
