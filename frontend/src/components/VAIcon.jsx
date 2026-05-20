import React from 'react';
import vaIcon from '../assets/va-icon.png';

/**
 * VAIcon – reusable VocalArmor wave logo icon.
 * @param {number}  size      – width & height in px (default 28)
 * @param {string}  className – extra CSS classes
 * @param {object}  style     – extra inline styles
 */
const VAIcon = ({ size = 28, className = '', style = {} }) => (
  <img
    src={vaIcon}
    alt="VocalArmor logo"
    width={size}
    height={size}
    className={className}
    style={{
      borderRadius: '6px',
      objectFit: 'cover',
      display: 'inline-block',
      flexShrink: 0,
      ...style,
    }}
  />
);

export default VAIcon;
